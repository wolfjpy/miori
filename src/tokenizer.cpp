#include "tokenizer.h"
#include "json.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace miori {

static void utf8_append(std::string& out, uint32_t cp) {
    if (cp <= 0x7F) out.push_back((char)cp);
    else if (cp <= 0x7FF) {
        out.push_back((char)(0xC0 | (cp >> 6)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back((char)(0xE0 | (cp >> 12)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else {
        out.push_back((char)(0xF0 | (cp >> 18)));
        out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    }
}

static uint32_t utf8_decode(const std::string& s, size_t i, size_t* len) {
    unsigned char c = (unsigned char)s[i];
    if (c < 0x80) { *len = 1; return c; }
    if ((c >> 5) == 0x6 && i + 1 < s.size()) { *len = 2; return ((c & 0x1F) << 6) | ((unsigned char)s[i + 1] & 0x3F); }
    if ((c >> 4) == 0xE && i + 2 < s.size()) { *len = 3; return ((c & 0x0F) << 12) | (((unsigned char)s[i + 1] & 0x3F) << 6) | ((unsigned char)s[i + 2] & 0x3F); }
    if ((c >> 3) == 0x1E && i + 3 < s.size()) { *len = 4; return ((c & 0x07) << 18) | (((unsigned char)s[i + 1] & 0x3F) << 12) | (((unsigned char)s[i + 2] & 0x3F) << 6) | ((unsigned char)s[i + 3] & 0x3F); }
    *len = 1; return c;
}

static bool is_space(uint32_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x00A0 || c == 0x3000;
}
static bool is_digit(uint32_t c) { return c >= '0' && c <= '9'; }
static bool is_letter(uint32_t c) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return true;
    if (c < 0x80) return false;
    if (is_space(c)) return false;
    // Treat non-ASCII non-space as letters (covers CJK, Cyrillic, Arabic, ...).
    // Emoji/punctuation may be grouped with words; acceptable for v0.1.
    return true;
}

static bool is_punct(uint32_t c) {
    if (c >= 0x80) return false;
    if (is_space(c) || is_letter(c) || is_digit(c)) return false;
    return true;
}

bool Tokenizer::load(const std::string& path, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open " + path; return false; }
    std::stringstream ss;
    ss << f.rdbuf();
    std::string data = ss.str();
    Json root;
    try {
        root = JsonParser::parse(data);
    } catch (const std::exception& e) {
        err = std::string("tokenizer json: ") + e.what();
        return false;
    }

    // byte -> unicode (GPT-2 byte-level BPE)
    std::vector<int> bs, cs;
    for (int b = 33; b <= 126; ++b) { bs.push_back(b); cs.push_back(b); }
    for (int b = 161; b <= 172; ++b) { bs.push_back(b); cs.push_back(b); }
    for (int b = 174; b <= 255; ++b) { bs.push_back(b); cs.push_back(b); }
    std::vector<bool> used(256, false);
    for (int b : bs) used[b] = true;
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        if (!used[b]) { bs.push_back(b); cs.push_back(256 + extra); ++extra; }
    }
    byte_to_uni_.resize(256);
    for (size_t i = 0; i < bs.size(); ++i) {
        utf8_append(byte_to_uni_[bs[i]], (uint32_t)cs[i]);
        uni_to_byte_[(uint32_t)cs[i]] = (uint8_t)bs[i];
    }

    const Json& model = root["model"];
    const Json& vocab = model["vocab"];
    int maxid = -1;
    for (const auto& kv : vocab.obj) maxid = std::max(maxid, (int)kv.second.as_int());
    id2tok_.assign((size_t)maxid + 1, std::string());
    for (const auto& kv : vocab.obj) {
        int id = (int)kv.second.as_int();
        vocab_[kv.first] = id;
        id2tok_[(size_t)id] = kv.first;
    }

    const Json& merges = model["merges"];
    ranks_.reserve(merges.arr.size() * 2);
    for (size_t i = 0; i < merges.arr.size(); ++i) {
        std::string a, b;
        const Json& e = merges.arr[i];
        if (e.is_array() && e.arr.size() >= 2) {
            a = e[0].str;
            b = e[1].str;
        } else if (e.is_string()) {
            const std::string& m = e.str;
            size_t sp = m.find(' ');
            if (sp == std::string::npos) continue;
            a = m.substr(0, sp);
            b = m.substr(sp + 1);
        } else {
            continue;
        }
        auto ia = vocab_.find(a), ib = vocab_.find(b);
        if (ia == vocab_.end() || ib == vocab_.end()) continue;
        uint64_t key = ((uint64_t)(uint32_t)ia->second << 32) | (uint32_t)ib->second;
        ranks_.emplace(key, (int)i);
    }

    if (root.contains("added_tokens")) {
        for (const auto& t : root["added_tokens"].arr) {
            int id = (int)t["id"].as_int();
            std::string content = t["content"].as_string();
            specials_.push_back({content, id});
            if ((size_t)id >= id2tok_.size()) id2tok_.resize((size_t)id + 1);
            if (id2tok_[(size_t)id].empty()) id2tok_[(size_t)id] = content;
        }
        std::sort(specials_.begin(), specials_.end(),
                  [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    }
    printf("tokenizer: vocab=%zu merges=%zu specials=%zu\n",
           vocab_.size(), ranks_.size(), specials_.size());
    return true;
}

const std::string& Tokenizer::token(int id) const {
    if (id < 0 || (size_t)id >= id2tok_.size()) throw std::runtime_error("bad token id");
    return id2tok_[(size_t)id];
}

int Tokenizer::id_of(const std::string& tok) const {
    auto it = vocab_.find(tok);
    return it == vocab_.end() ? -1 : it->second;
}

void Tokenizer::pretokenize(const std::string& text, std::vector<std::pair<size_t, size_t>>& spans) const {
    const size_t n = text.size();
    size_t i = 0;
    while (i < n) {
        size_t len = 0;
        uint32_t c = utf8_decode(text, i, &len);
        // contractions
        if (c == '\'' && i + 1 < n) {
            size_t l2 = 0;
            uint32_t c2 = utf8_decode(text, i + 1, &l2);
            uint32_t lo = (c2 >= 'A' && c2 <= 'Z') ? c2 + 32 : c2;
            if (lo == 's' || lo == 't' || lo == 'm' || lo == 'd') {
                spans.push_back({i, i + 1 + l2}); i += 1 + l2; continue;
            }
            if (i + 2 < n) {
                size_t l3 = 0;
                uint32_t c3 = utf8_decode(text, i + 2, &l3);
                uint32_t lo3 = (c3 >= 'A' && c3 <= 'Z') ? c3 + 32 : c3;
                if ((lo == 'r' && lo3 == 'e') || (lo == 'v' && lo3 == 'e') || (lo == 'l' && lo3 == 'l')) {
                    spans.push_back({i, i + 2 + l3}); i += 2 + l3; continue;
                }
            }
        }
        if (is_letter(c)) {
            size_t j = i + len;
            while (j < n) {
                size_t l = 0; uint32_t d = utf8_decode(text, j, &l);
                if (!is_letter(d)) break;
                j += l;
            }
            spans.push_back({i, j}); i = j; continue;
        }
        if (is_digit(c)) {
            size_t j = i + len;
            int count = 1;
            while (j < n && count < 3) {
                size_t l = 0; uint32_t d = utf8_decode(text, j, &l);
                if (!is_digit(d)) break;
                j += l; ++count;
            }
            spans.push_back({i, j}); i = j; continue;
        }
        if (is_space(c)) {
            size_t j = i;
            // newline run
            if (c == '\n' || c == '\r') {
                while (j < n) {
                    size_t l = 0; uint32_t d = utf8_decode(text, j, &l);
                    if (!is_space(d)) break;
                    j += l;
                }
                spans.push_back({i, j}); i = j; continue;
            }
            while (j < n) {
                size_t l = 0; uint32_t d = utf8_decode(text, j, &l);
                if (!is_space(d) || d == '\n' || d == '\r') break;
                j += l;
            }
            size_t k = j;
            if (k < n) {
                size_t l = 0; uint32_t d = utf8_decode(text, k, &l);
                if (is_letter(d)) {
                    size_t e = k + l;
                    while (e < n) {
                        size_t l2 = 0; uint32_t e2 = utf8_decode(text, e, &l2);
                        if (!is_letter(e2)) break;
                        e += l2;
                    }
                    spans.push_back({i, e}); i = e; continue;
                }
                if (is_punct(d)) {
                    size_t e = k + l;
                    while (e < n) {
                        size_t l2 = 0; uint32_t e2 = utf8_decode(text, e, &l2);
                        if (!is_punct(e2) && e2 != '\n' && e2 != '\r') break;
                        e += l2;
                    }
                    spans.push_back({i, e}); i = e; continue;
                }
            }
            spans.push_back({i, j}); i = j; continue;
        }
        // punctuation/symbols, optional leading space handled above
        {
            size_t j = i + len;
            while (j < n) {
                size_t l = 0; uint32_t d = utf8_decode(text, j, &l);
                if (!(is_punct(d) || d == '\n' || d == '\r')) break;
                j += l;
            }
            spans.push_back({i, j}); i = j; continue;
        }
    }
}

void Tokenizer::bpe(const std::string& span, std::vector<int>& out) const {
    std::vector<int> ids;
    ids.reserve(span.size());
    for (unsigned char b : span) {
        auto it = vocab_.find(byte_to_uni_[b]);
        if (it == vocab_.end()) throw std::runtime_error("byte not in vocab");
        ids.push_back(it->second);
    }
    while (ids.size() > 1) {
        int best = INT32_MAX, besti = -1;
        for (size_t i = 0; i + 1 < ids.size(); ++i) {
            uint64_t key = ((uint64_t)(uint32_t)ids[i] << 32) | (uint32_t)ids[i + 1];
            auto it = ranks_.find(key);
            if (it != ranks_.end() && it->second < best) { best = it->second; besti = (int)i; }
        }
        if (besti < 0) break;
        std::vector<int> merged;
        merged.reserve(ids.size());
        for (size_t i = 0; i < ids.size();) {
            if (i + 1 < ids.size() &&
                ((uint64_t)(uint32_t)ids[i] << 32 | (uint32_t)ids[i + 1]) ==
                    (((uint64_t)(uint32_t)ids[besti] << 32) | (uint32_t)ids[besti + 1])) {
                const std::string s = id2tok_[(size_t)ids[i]] + id2tok_[(size_t)ids[i + 1]];
                auto it = vocab_.find(s);
                if (it == vocab_.end()) { merged.push_back(ids[i]); ++i; continue; }
                merged.push_back(it->second);
                i += 2;
            } else {
                merged.push_back(ids[i]); ++i;
            }
        }
        ids.swap(merged);
    }
    out.insert(out.end(), ids.begin(), ids.end());
}

void Tokenizer::encode_into(const std::string& text, std::vector<int>& out) const {
    std::vector<std::pair<size_t, size_t>> spans;
    pretokenize(text, spans);
    for (const auto& sp : spans) bpe(text.substr(sp.first, sp.second - sp.first), out);
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> out;
    encode_into(text, out);
    return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string bytes;
    for (int id : ids) {
        if (id < 0 || (size_t)id >= id2tok_.size()) continue;
        const std::string& t = id2tok_[(size_t)id];
        if (id >= 124893) { bytes += t; continue; }
        size_t i = 0;
        while (i < t.size()) {
            size_t len = 0;
            uint32_t cp = utf8_decode(t, i, &len);
            auto it = uni_to_byte_.find(cp);
            if (it != uni_to_byte_.end()) bytes.push_back((char)it->second);
            i += len;
        }
    }
    return bytes;
}

std::vector<int> Tokenizer::chat(const std::string& system, const std::string& user,
                                 bool thinking) const {
    std::vector<int> ids;
    ids.push_back(124894);  // <|startoftext|>
    auto add_special = [&](int id, const std::string& s) {
        ids.push_back(id);
        encode_into(s, ids);
    };
    auto add_end = [&]() {
        ids.push_back(124900);  // <|im_end|>
        encode_into("\n", ids);
    };
    if (!system.empty()) {
        add_special(124899, "system\n" + system);
        add_end();
    }
    add_special(124899, "user\n" + user);
    add_end();
    ids.push_back(124899);
    encode_into("assistant\n", ids);
    if (thinking) {
        ids.push_back(124901);  // <think>
    } else {
        ids.push_back(124901);
        ids.push_back(124902);  // </think>
    }
    return ids;
}

}  // namespace miori

namespace miori {
std::string Tokenizer::decode_plain(const std::vector<int>& ids) const {
    std::vector<int> text;
    text.reserve(ids.size());
    for (int id : ids) if (id < 124893) text.push_back(id);
    return decode(text);
}
}  // namespace miori
