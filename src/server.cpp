#include "server.h"
#include "engine.h"
#include "tokenizer.h"
#include "json.h"
#include "batch.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace miori {

static std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
                else o += (char)c;
        }
    }
    return o;
}

static bool send_all(int fd, const std::string& s) {
    size_t off = 0;
    while (off < s.size()) {
        ssize_t n = ::send(fd, s.data() + off, s.size() - off, 0);
        if (n <= 0) return false;
        off += (size_t)n;
    }
    return true;
}

static std::string read_request(int fd, std::string& body) {
    std::string req;
    char buf[8192];
    size_t hdr = std::string::npos;
    while (hdr == std::string::npos) {
        ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) return {};
        req.append(buf, (size_t)n);
        hdr = req.find("\r\n\r\n");
        if (req.size() > 4 * 1024 * 1024) return {};
    }
    size_t clen = 0;
    std::string lower = req.substr(0, hdr);
    for (auto& c : lower) c = (char)tolower((unsigned char)c);
    size_t p = lower.find("content-length:");
    if (p != std::string::npos) clen = (size_t)atoi(req.c_str() + p + 15);
    body = req.substr(hdr + 4);
    while (body.size() < clen) {
        ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        body.append(buf, (size_t)n);
    }
    return req.substr(0, hdr);
}

struct Job {
    std::vector<int> prompt;
    int max_tokens = 16;
    std::vector<int> out;
    bool done = false;
    std::mutex m;
    std::condition_variable cv;
};

class Scheduler {
public:
    explicit Scheduler(BatchEngine& be, std::mutex& eng_mu) : be_(be), eng_mu_(eng_mu) {}
    void submit(const std::shared_ptr<Job>& j) {
        { std::lock_guard<std::mutex> lk(mu_); q_.push_back(j); }
        cv_.notify_one();
    }
    void run() {
        for (;;) {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return !q_.empty(); });
            cv_.wait_for(lk, std::chrono::milliseconds(5), [&] { return q_.size() >= (size_t)BatchEngine::kBmax; });
            std::vector<std::shared_ptr<Job>> batch;
            while (batch.size() < (size_t)BatchEngine::kBmax && !q_.empty()) {
                batch.push_back(q_.front());
                q_.pop_front();
            }
            lk.unlock();
            std::vector<int> slots;
            slots.reserve(batch.size());
            {
                std::lock_guard<std::mutex> el(eng_mu_);
                for (auto& j : batch) slots.push_back(be_.add_sequence(j->prompt, j->max_tokens));
            }
            while (be_.running() > 0) be_.decode_step();
            for (size_t i = 0; i < batch.size(); ++i) {
                if (slots[i] >= 0) batch[i]->out = be_.generated(slots[i]);
                { std::lock_guard<std::mutex> jl(batch[i]->m); batch[i]->done = true; }
                batch[i]->cv.notify_all();
            }
            be_.clear_slots();
        }
    }
private:
    BatchEngine& be_;
    std::mutex& eng_mu_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Job>> q_;
};
struct PrefixCache {
    std::string system;
    bool valid = false;
};

static void make_prefix(Tokenizer& t, const std::string& system, std::vector<int>& v) {
    v.push_back(124894);  // <|startoftext|>
    if (!system.empty()) {
        v.push_back(124899);  // <|im_start|>
        auto a = t.encode("system\n" + system);
        v.insert(v.end(), a.begin(), a.end());
        v.push_back(124900);  // <|im_end|>
        auto nl = t.encode("\n");
        v.insert(v.end(), nl.begin(), nl.end());
    }
}

static void make_suffix(Tokenizer& t, const std::string& user, bool thinking, std::vector<int>& v) {
    v.push_back(124899);
    auto b = t.encode("user\n" + user);
    v.insert(v.end(), b.begin(), b.end());
    v.push_back(124900);
    auto nl = t.encode("\n");
    v.insert(v.end(), nl.begin(), nl.end());
    v.push_back(124899);
    auto c = t.encode("assistant\n");
    v.insert(v.end(), c.begin(), c.end());
    v.push_back(124901);  // <think>
    if (!thinking) v.push_back(124902);  // </think>
}
static std::atomic<long> g_requests{0};
static std::atomic<long> g_tokens{0};

static void handle(int fd, Engine& eng, Tokenizer& tok, std::mutex& mu, PrefixCache& pc, Scheduler& sched) {
    std::string body;
    std::string header = read_request(fd, body);
    if (header.empty()) { ::close(fd); return; }
    if (header.find("GET /metrics") != std::string::npos) {
        std::string b = "# HELP miori_requests_total Requests served\n"
                        "# TYPE miori_requests_total counter\n"
                        "miori_requests_total " + std::to_string(g_requests.load()) + "\n"
                        "# HELP miori_output_tokens_total Output tokens generated\n"
                        "# TYPE miori_output_tokens_total counter\n"
                        "miori_output_tokens_total " + std::to_string(g_tokens.load()) + "\n";
        std::string r = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: " +
                        std::to_string(b.size()) + "\r\nConnection: close\r\n\r\n" + b;
        send_all(fd, r);
        ::close(fd);
        return;
    }
    if (header.find("GET /health") != std::string::npos) {
        std::string b = "{\"status\":\"ok\"}";
        std::string r = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " +
                        std::to_string(b.size()) + "\r\nConnection: close\r\n\r\n" + b;
        send_all(fd, r);
        ::close(fd);
        return;
    }
    if (header.find("GET /v1/models") != std::string::npos) {
        std::string b = "{\"object\":\"list\",\"data\":[{\"id\":\"lfm3-vl-2b-mlx-4bit\",\"object\":\"model\"}]}";
        std::string r = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " +
                        std::to_string(b.size()) + "\r\nConnection: close\r\n\r\n" + b;
        send_all(fd, r);
        ::close(fd);
        return;
    }
    if (header.find("POST /v1/chat/completions") == std::string::npos) {
        std::string r = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        send_all(fd, r);
        ::close(fd);
        return;
    }
    Json root;
    try {
        root = JsonParser::parse(body);
    } catch (...) {
        std::string r = "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        send_all(fd, r);
        ::close(fd);
        return;
    }
    std::string system, user;
    if (root.contains("messages")) {
        for (const auto& m : root["messages"].arr) {
            std::string role = m.contains("role") ? m["role"].str : "user";
            if (!m.contains("content")) continue;
            const Json& c = m["content"];
            std::string text;
            if (c.is_string()) text = c.str;
            else if (c.is_array()) {
                for (const auto& part : c.arr)
                    if (part.contains("text") && part["text"].is_string()) text += part["text"].str;
            }
            if (role == "system") system = text;
            else if (role == "user") user += text;
            else if (role == "assistant") { /* history not supported in v0.1 */ }
        }
    }
    bool stream = root.contains("stream") && root["stream"].boolean;
    int max_tokens = root.contains("max_tokens") ? (int)root["max_tokens"].as_int() : 256;
    float temp = root.contains("temperature") ? (float)root["temperature"].as_double() : 0.1f;
    bool thinking = false;
    if (root.contains("chat_template_kwargs") && root["chat_template_kwargs"].contains("enable_thinking"))
        thinking = root["chat_template_kwargs"]["enable_thinking"].boolean;

    std::vector<int> out;
    size_t prompt_tokens = 0;
    GenConfig cfg;
    cfg.max_tokens = max_tokens;
    cfg.temperature = temp;
    if (temp <= 0.01f) {
        // Greedy requests go through the 4-slot batch decoder.
        auto job = std::make_shared<Job>();
        job->prompt = tok.chat(system, user, thinking);
        job->max_tokens = max_tokens;
        prompt_tokens = job->prompt.size();
        sched.submit(job);
        std::unique_lock<std::mutex> jl(job->m);
        job->cv.wait(jl, [&] { return job->done; });
        out = job->out;
        if (out.empty()) {
            std::lock_guard<std::mutex> lock(mu);
            out = eng.generate(job->prompt, cfg);
        }
    } else {
        // Sampling requests keep the single-sequence path (and its prefix cache).
        std::vector<int> prefix, suffix;
        make_prefix(tok, system, prefix);
        make_suffix(tok, user, thinking, suffix);
        prompt_tokens = prefix.size() + suffix.size();
        std::lock_guard<std::mutex> lock(mu);
        if (!pc.valid || pc.system != system) {
            eng.reset();
            eng.prefill_ids(prefix, false);
            eng.snapshot();
            pc.system = system;
            pc.valid = true;
        } else {
            eng.restore();
        }
        out = eng.generate_from_state(suffix, cfg);
    }
    g_requests.fetch_add(1);
    g_tokens.fetch_add((long)out.size());
    std::string text = tok.decode_plain(out);
    std::string id = "chatcmpl-miori";
    if (stream) {
        std::string h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                        "Cache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n";
        if (!send_all(fd, h)) { ::close(fd); return; }
        // Emit in small chunks so clients see a stream; the engine itself is
        // synchronous in v0.1.
        size_t step = 8;
        for (size_t i = 0; i < text.size(); i += step) {
            std::string piece = text.substr(i, step);
            std::string ev = "data: {\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\","
                             "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" +
                             json_escape(piece) + "\"},\"finish_reason\":null}]}\n\n";
            if (!send_all(fd, ev)) break;
            usleep(2000);
        }
        send_all(fd, "data: {\"id\":\"" + id + "\",\"choices\":[{\"index\":0,\"delta\":{},"
                     "\"finish_reason\":\"stop\"}]}\n\n");
        send_all(fd, "data: [DONE]\n\n");
    } else {
        std::string resp = "{\"id\":\"" + id + "\",\"object\":\"chat.completion\",\"created\":0,"
                           "\"model\":\"lfm3-vl-2b-mlx-4bit\",\"choices\":[{\"index\":0,"
                           "\"message\":{\"role\":\"assistant\",\"content\":\"" +
                           json_escape(text) + "\"},\"finish_reason\":\"stop\"}],"
                           "\"usage\":{\"prompt_tokens\":" + std::to_string(prompt_tokens) +
                           ",\"completion_tokens\":" + std::to_string(out.size()) +
                           ",\"total_tokens\":" + std::to_string(prompt_tokens + out.size()) + "}}";
        std::string r = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " +
                        std::to_string(resp.size()) + "\r\nConnection: close\r\n\r\n" + resp;
        send_all(fd, r);
    }
    ::close(fd);
}

int run_server(Metal& m, Weights& w, int port) {
    Tokenizer tok;
    std::string err;
    if (!tok.load("models/tokenizer.json", err)) {
        fprintf(stderr, "tokenizer: %s\n", err.c_str());
        return 1;
    }
    Engine eng(m, w);
    if (!eng.init(err)) {
        fprintf(stderr, "engine: %s\n", err.c_str());
        return 1;
    }
    std::mutex mu;
    BatchEngine be(m, w, eng);
    if (!be.init(err)) {
        fprintf(stderr, "batch engine: %s\n", err.c_str());
        return 1;
    }
    be.set_kernels(BatchEngine::Kernels::Gemv4);
    Scheduler sched(be, mu);
    std::thread([&sched]() { sched.run(); }).detach();

    int srv = ::socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { perror("socket"); return 1; }
    int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((uint16_t)port);
    if (bind(srv, (sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }
    if (listen(srv, 128) < 0) { perror("listen"); return 1; }
    printf("miori serving on http://127.0.0.1:%d/v1/chat/completions\n", port);
    PrefixCache pc;
    std::atomic<int> live{0};
    for (;;) {
        int fd = ::accept(srv, nullptr, nullptr);
        if (fd < 0) continue;
        live.fetch_add(1);
        std::thread([fd, &eng, &tok, &mu, &pc, &sched, &live]() {
            handle(fd, eng, tok, mu, pc, sched);
            live.fetch_sub(1);
        }).detach();
    }
}

}  // namespace miori
