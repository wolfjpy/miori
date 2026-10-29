#include "server.h"
#include "engine.h"
#include "tokenizer.h"
#include "json.h"
#include "batch.h"
#include "vision.h"
#include "m6.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <fcntl.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <unordered_map>
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

static std::string b64decode(const std::string& in) {
    static const char* tab = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    int rev[256];
    for (int i = 0; i < 256; ++i) rev[i] = -1;
    for (int i = 0; i < 64; ++i) rev[(unsigned char)tab[i]] = i;
    std::string out;
    int val = 0, bits = 0;
    for (unsigned char c : in) {
        if (c == '=') break;
        int d = rev[c];
        if (d < 0) continue;
        val = (val << 6) | d;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((char)((val >> bits) & 0xFF));
        }
    }
    return out;
}

static std::vector<int> image_prompt(const Tokenizer& t, const std::string& system,
                                     const std::string& user, const std::vector<int>& counts, bool thinking) {
    std::vector<int> ids;
    ids.push_back(124894);
    const std::string sys = system.empty() ? "You are a helpful assistant." : system;
    ids.push_back(124899);
    auto a = t.encode("system\n" + sys);
    ids.insert(ids.end(), a.begin(), a.end());
    ids.push_back(124900);
    auto nl = t.encode("\n");
    ids.insert(ids.end(), nl.begin(), nl.end());
    ids.push_back(124899);
    auto b = t.encode("user\n");
    ids.insert(ids.end(), b.begin(), b.end());
    for (int n_image : counts) {
        ids.push_back(miori::kVisionStartId);
        for (int i = 0; i < n_image; ++i) ids.push_back(miori::kImageId);
        ids.push_back(miori::kVisionEndId);
    }
    auto c = t.encode(user);
    ids.insert(ids.end(), c.begin(), c.end());
    ids.push_back(124900);
    ids.insert(ids.end(), nl.begin(), nl.end());
    ids.push_back(124899);
    auto d = t.encode("assistant\n");
    ids.insert(ids.end(), d.begin(), d.end());
    ids.push_back(124901);
    if (!thinking) ids.push_back(124902);
    return ids;
}
class ImageCache {
public:
    bool get(uint64_t key, std::vector<uint16_t>& feats, int& n_tokens) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = map_.find(key);
        if (it == map_.end()) return false;
        feats = it->second.first;
        n_tokens = it->second.second;
        return true;
    }
    void put(uint64_t key, const std::vector<uint16_t>& feats, int n_tokens) {
        std::lock_guard<std::mutex> lk(mu_);
        if (map_.size() >= 32) {
            map_.erase(lru_.front());
            lru_.pop_front();
        }
        map_[key] = {feats, n_tokens};
        lru_.push_back(key);
    }
private:
    std::mutex mu_;
    std::unordered_map<uint64_t, std::pair<std::vector<uint16_t>, int>> map_;
    std::deque<uint64_t> lru_;
};
static ImageCache g_images;
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
    bool submit(const std::shared_ptr<Job>& j) {
        { std::lock_guard<std::mutex> lk(mu_); if (q_.size() >= 256) return false; q_.push_back(j); }
        cv_.notify_one();
        return true;
    }
    size_t depth() { std::lock_guard<std::mutex> lk(mu_); return q_.size(); }
    int backlog() { std::lock_guard<std::mutex> lk(mu_); return (int)q_.size() + active_cnt_.load(); }
    void run() {
        std::map<int, std::shared_ptr<Job>> active;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu_);
                if (q_.empty() && active.empty())
                    cv_.wait(lk, [&] { return !q_.empty(); });
                if (!q_.empty() && active.size() < (size_t)BatchEngine::kBmax)
                    cv_.wait_for(lk, std::chrono::milliseconds(5),
                                 [&] { return q_.size() >= (size_t)BatchEngine::kBmax; });
            }
            // Fill free slots from the queue.
            for (;;) {
                std::shared_ptr<Job> j;
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    if (q_.empty()) break;
                    j = q_.front();
                }
                int s = -1;
                {
                    std::lock_guard<std::mutex> el(eng_mu_);
                    s = be_.add_sequence(j->prompt, j->max_tokens);
                }
                if (s < 0) break;
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    if (!q_.empty() && q_.front() == j) q_.pop_front();
                }
                active[s] = j;
                active_cnt_.fetch_add(1);
            }
            if (active.empty()) {
                // Unserviceable request (e.g. prompt too long): drop it instead of spinning.
                std::shared_ptr<Job> j;
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    if (!q_.empty()) { j = q_.front(); q_.pop_front(); }
                }
                if (j) {
                    { std::lock_guard<std::mutex> jl(j->m); j->done = true; }
                    j->cv.notify_all();
                }
                continue;
            }
            be_.decode_step();
            std::vector<int> done;
            for (auto& kv : active)
                if (be_.finished(kv.first)) done.push_back(kv.first);
            for (int s : done) {
                auto j = active[s];
                j->out = be_.generated(s);
                be_.release_slot(s);
                { std::lock_guard<std::mutex> jl(j->m); j->done = true; }
                j->cv.notify_all();
                active.erase(s);
                active_cnt_.fetch_sub(1);
            }
        }
    }
private:
    BatchEngine& be_;
    std::mutex& eng_mu_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Job>> q_;
    std::atomic<int> active_cnt_{0};
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

static void handle(int fd, Engine& eng, Tokenizer& tok, std::mutex& mu, PrefixCache& pc, Scheduler& sched, Vision* vis) {
    std::string body;
    std::string header = read_request(fd, body);
    if (header.empty()) { ::close(fd); return; }
    if (header.find("GET /metrics") != std::string::npos) {
        std::string b = "# HELP miori_requests_total Requests served\n"
                        "# TYPE miori_requests_total counter\n"
                        "miori_requests_total " + std::to_string(g_requests.load()) + "\n"
                        "# HELP miori_output_tokens_total Output tokens generated\n"
                        "# TYPE miori_output_tokens_total counter\n"
                        "miori_output_tokens_total " + std::to_string(g_tokens.load()) + "\n"
                        "# HELP miori_queue_depth Queued requests\n"
                        "# TYPE miori_queue_depth gauge\n"
                        "miori_queue_depth " + std::to_string(sched.depth()) + "\n";
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
    std::vector<std::string> image_urls;
    if (root.contains("messages")) {
        for (const auto& m : root["messages"].arr) {
            std::string role = m.contains("role") ? m["role"].str : "user";
            if (!m.contains("content")) continue;
            const Json& c = m["content"];
            std::string text;
            if (c.is_string()) text = c.str;
            else if (c.is_array()) {
                for (const auto& part : c.arr) {
                    if (part.contains("text") && part["text"].is_string()) text += part["text"].str;
                    else if (part.contains("image_url") && part["image_url"].contains("url"))
                        image_urls.push_back(part["image_url"]["url"].str);
                }
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
    if (!image_urls.empty() && vis) {
        std::vector<uint16_t> feats;
        std::vector<int> counts;
        for (const auto& url : image_urls) {
            const size_t comma = url.find(',');
            std::string b64 = comma == std::string::npos ? url : url.substr(comma + 1);
            std::string bytes = b64decode(b64);
            const uint64_t key = std::hash<std::string>{}(bytes) ^ ((uint64_t)bytes.size() << 32);
            std::vector<uint16_t> f;
            int nt = 0;
            if (g_images.get(key, f, nt)) {
                counts.push_back(nt);
                feats.insert(feats.end(), f.begin(), f.end());
                continue;
            }
            char path[] = "/tmp/miori_imgXXXXXX";
            int fdt = mkstemp(path);
            if (fdt < 0 || bytes.empty()) {
                std::string r = "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                send_all(fd, r); ::close(fd); return;
            }
            (void)::write(fdt, bytes.data(), bytes.size());
            ::close(fdt);
            std::vector<uint16_t> patches;
            ImageInfo info;
            std::string ierr;
            bool ok = preprocess_image(path, 65536, 1048576, patches, info, ierr);
            if (ok) ok = vis->encode(patches, info, f, ierr);
            ::unlink(path);
            if (!ok) {
                fprintf(stderr, "vision: %s\n", ierr.c_str());
                std::string r = "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                send_all(fd, r); ::close(fd); return;
            }
            g_images.put(key, f, info.n_tokens);
            counts.push_back(info.n_tokens);
            feats.insert(feats.end(), f.begin(), f.end());
        }
        auto ids = image_prompt(tok, system, user, counts, thinking);
        if (ids.size() >= 4000) {
            std::string r = "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            send_all(fd, r); ::close(fd); return;
        }
        MTL::Buffer* emb = eng.alloc_temp((size_t)ids.size() * 2048 * 2);
        std::vector<uint16_t> host((size_t)ids.size() * 2048);
        {
            std::lock_guard<std::mutex> lock(mu);
            eng.embed_tokens(ids.data(), (int)ids.size(), emb);
        }
        memcpy(host.data(), emb->contents(), host.size() * 2);
        emb->release();
        int spliced = 0;
        const int total_feats = (int)(feats.size() / 2048);
        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] != miori::kImageId) continue;
            if (spliced >= total_feats) break;
            memcpy(&host[i * 2048], &feats[(size_t)spliced * 2048], 2048 * 2);
            ++spliced;
        }
        if (spliced != total_feats) {
            std::string r = "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            send_all(fd, r); ::close(fd); return;
        }
        {
            std::lock_guard<std::mutex> lock(mu);
            eng.reset();
            eng.prefill_embeds(host.data(), (int)ids.size(), true);
            out = eng.decode_loop(cfg);
        }
        prompt_tokens = ids.size();
    } else if (temp <= 0.01f) {
    } else if (temp <= 0.01f) {
        // Greedy requests go through the 4-slot batch decoder.
        auto job = std::make_shared<Job>();
        job->prompt = tok.chat(system, user, thinking);
        job->max_tokens = max_tokens;
        prompt_tokens = job->prompt.size();
        if (!sched.submit(job)) {
            std::string r = "HTTP/1.1 429 Too Many Requests\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            send_all(fd, r);
            ::close(fd);
            return;
        }
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
    Vision vis(m, w);
    bool vis_ok = vis.init(err);
    if (!vis_ok) fprintf(stderr, "vision disabled: %s\n", err.c_str());
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
        std::thread([fd, &eng, &tok, &mu, &pc, &sched, &vis, vis_ok, &live]() {
            handle(fd, eng, tok, mu, pc, sched, vis_ok ? &vis : nullptr);
            live.fetch_sub(1);
        }).detach();
    }
}

}  // namespace miori
