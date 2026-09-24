// obs_dump: read CLRJ states, rebuild each position in the C++ engine
// (SetState) and write its 18x9x9 observation and 6561 legal mask as raw
// float32 — to compare the deploy/eval observation with the training one.
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "canonical.h"
#include "game.h"

int main(int argc, char** argv) {
  if (argc < 4) { std::printf("usage: obs_dump states.bin obs.f32 legal.u8 [canon views.i32]\n"); return 2; }
  const bool canon = argc > 5 && std::string(argv[4]) == "canon";
  FILE* fv = canon ? std::fopen(argv[5], "wb") : nullptr;
  const clines::D4Maps d4;
  std::ifstream f(argv[1], std::ios::binary);
  char magic[4]; f.read(magic, 4);
  if (!f || std::string(magic, 4) != "CLRJ") { std::printf("bad states file\n"); return 1; }
  int32_t n = 0; f.read(reinterpret_cast<char*>(&n), 4);
  FILE* fo = std::fopen(argv[2], "wb");
  FILE* fl = std::fopen(argv[3], "wb");
  std::vector<float> obs(18 * clines::kNN), legal(clines::kActions);
  std::vector<uint8_t> legal8(clines::kActions);
  for (int i = 0; i < n; ++i) {
    int8_t board[81]; f.read(reinterpret_cast<char*>(board), 81);
    int32_t k = 0; f.read(reinterpret_cast<char*>(&k), 4);
    std::vector<clines::NextBall> nb;
    for (int t = 0; t < 3; ++t) {
      int32_t r, c, col;
      f.read(reinterpret_cast<char*>(&r), 4); f.read(reinterpret_cast<char*>(&c), 4);
      f.read(reinterpret_cast<char*>(&col), 4);
      if (t < k) nb.push_back({(int)r, (int)c, (int)col});
    }
    char tail[12]; f.read(tail, 12);
    clines::Game g(0);
    if (canon) {
      const clines::Canonical cf = clines::Canonicalize(board, nb, d4);
      g.SetState(cf.board, cf.next, 0, 0);
      const int32_t view = cf.view;
      std::fwrite(&view, sizeof(int32_t), 1, fv);
    } else {
      g.SetState(board, nb, 0, 0);
    }
    g.BuildObs(obs.data());
    g.LegalMask(legal.data());
    for (int a = 0; a < clines::kActions; ++a) legal8[a] = legal[a] > 0.5f;
    std::fwrite(obs.data(), sizeof(float), obs.size(), fo);
    std::fwrite(legal8.data(), 1, legal8.size(), fl);
  }
  std::fclose(fo); std::fclose(fl);
  if (fv) std::fclose(fv);
  std::printf("dumped %d states\n", n);
  return 0;
}
