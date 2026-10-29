// Minimal JSON DOM + parser (C++20, no dependencies). Used for safetensors headers,
// tokenizer.json and the HTTP API.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace miori {

struct Json {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string str;
    std::vector<Json> arr;
    std::map<std::string, Json> obj;

    bool is_null() const { return type == Type::Null; }
    bool is_object() const { return type == Type::Object; }
    bool is_array() const { return type == Type::Array; }
    bool is_string() const { return type == Type::String; }
    bool is_number() const { return type == Type::Number; }

    const Json& operator[](const std::string& k) const {
        auto it = obj.find(k);
        if (it == obj.end()) throw std::runtime_error("json: missing key " + k);
        return it->second;
    }
    const Json& operator[](size_t i) const { return arr.at(i); }
    bool contains(const std::string& k) const { return obj.find(k) != obj.end(); }

    std::string as_string() const {
        if (type != Type::String) throw std::runtime_error("json: not a string");
        return str;
    }
    int64_t as_int() const {
        if (type != Type::Number) throw std::runtime_error("json: not a number");
        return (int64_t)number;
    }
    double as_double() const {
        if (type != Type::Number) throw std::runtime_error("json: not a number");
        return number;
    }
    size_t size() const { return type == Type::Array ? arr.size() : obj.size(); }
};

class JsonParser {
public:
    static Json parse(std::string_view text) {
        JsonParser p(text);
        Json v = p.value();
        p.ws();
        return v;
    }

private:
    std::string_view s_;
    size_t i_ = 0;
    explicit JsonParser(std::string_view s) : s_(s) {}

    [[noreturn]] void fail(const char* msg) const {
        throw std::runtime_error(std::string("json: ") + msg + " at " + std::to_string(i_));
    }
    void ws() {
        while (i_ < s_.size()) {
            char c = s_[i_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { ++i_; continue; }
            break;
        }
    }
    char peek() const { return i_ < s_.size() ? s_[i_] : '\0'; }
    char take() {
        if (i_ >= s_.size()) fail("eof");
        return s_[i_++];
    }
    void expect(char c) {
        if (take() != c) fail("unexpected char");
    }

    Json value() {
        ws();
        char c = peek();
        switch (c) {
            case '{': return object();
            case '[': return array();
            case '"': { Json v; v.type = Json::Type::String; v.str = string(); return v; }
            case 't': case 'f': return boolean();
            case 'n': return null();
            default: return number();
        }
    }

    Json object() {
        Json v; v.type = Json::Type::Object;
        expect('{');
        ws();
        if (peek() == '}') { ++i_; return v; }
        for (;;) {
            ws();
            std::string k = string();
            ws(); expect(':');
            v.obj.emplace(std::move(k), value());
            ws();
            char c = take();
            if (c == '}') break;
            if (c != ',') fail("expected , or }");
        }
        return v;
    }

    Json array() {
        Json v; v.type = Json::Type::Array;
        expect('[');
        ws();
        if (peek() == ']') { ++i_; return v; }
        for (;;) {
            v.arr.push_back(value());
            ws();
            char c = take();
            if (c == ']') break;
            if (c != ',') fail("expected , or ]");
        }
        return v;
    }

    Json boolean() {
        Json v; v.type = Json::Type::Bool;
        if (s_.substr(i_, 4) == "true") { v.boolean = true; i_ += 4; }
        else if (s_.substr(i_, 5) == "false") { v.boolean = false; i_ += 5; }
        else fail("bad bool");
        return v;
    }

    Json null() {
        if (s_.substr(i_, 4) != "null") fail("bad null");
        i_ += 4;
        return Json{};
    }

    Json number() {
        Json v; v.type = Json::Type::Number;
        size_t start = i_;
        if (peek() == '-') ++i_;
        while (isdigit((unsigned char)peek())) ++i_;
        if (peek() == '.') { ++i_; while (isdigit((unsigned char)peek())) ++i_; }
        if (peek() == 'e' || peek() == 'E') {
            ++i_;
            if (peek() == '+' || peek() == '-') ++i_;
            while (isdigit((unsigned char)peek())) ++i_;
        }
        if (start == i_) fail("bad number");
        v.number = strtod(std::string(s_.substr(start, i_ - start)).c_str(), nullptr);
        return v;
    }

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

    uint32_t hex4() {
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            char c = take();
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
            else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
            else fail("bad \\u");
        }
        return v;
    }

    std::string string() {
        expect('"');
        std::string out;
        for (;;) {
            char c = take();
            if (c == '"') break;
            if (c == '\\') {
                char e = take();
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        uint32_t cp = hex4();
                        if (cp >= 0xD800 && cp <= 0xDBFF && s_.substr(i_, 2) == "\\u") {
                            i_ += 2;
                            uint32_t lo = hex4();
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        }
                        utf8_append(out, cp);
                        break;
                    }
                    default: fail("bad escape");
                }
            } else {
                out.push_back(c);
            }
        }
        return out;
    }
};

}  // namespace miori
