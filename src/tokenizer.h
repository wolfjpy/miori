#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace miori {

class Tokenizer {
public:
    bool load(const std::string& path, std::string& err);
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;
    std::string decode_plain(const std::vector<int>& ids) const;
    std::vector<int> chat(const std::string& system, const std::string& user,
                          bool thinking) const;
    const std::string& token(int id) const;
    int id_of(const std::string& tok) const;
    size_t vocab_size() const { return id2tok_.size(); }

private:
    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string> id2tok_;
    std::unordered_map<uint64_t, int> ranks_;
    std::vector<std::string> byte_to_uni_;   // 256 entries
    std::unordered_map<uint32_t, uint8_t> uni_to_byte_;
    std::vector<std::pair<std::string, int>> specials_;

    void pretokenize(const std::string& text, std::vector<std::pair<size_t, size_t>>& spans) const;
    void bpe(const std::string& span, std::vector<int>& out) const;
    void encode_into(const std::string& text, std::vector<int>& out) const;
};

}  // namespace miori
