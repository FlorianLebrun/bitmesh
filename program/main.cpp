#include "./gates_unit/GateObject.h"
#include <functional>
#include <stdio.h>
#include <windows.h>

using namespace ins;


extern void print_clean() {
   CONSOLE_SCREEN_BUFFER_INFO csbi;
   HANDLE hStdOut = GetStdHandle(STD_OUTPUT_HANDLE);
   if (GetConsoleScreenBufferInfo(hStdOut, &csbi)) {
      COORD coord;
      coord.X = 0;
      coord.Y = 0;
      while (coord.Y < csbi.dwSize.Y) {
         DWORD count;
         FillConsoleOutputCharacterA(hStdOut, ' ', csbi.dwSize.X, coord, &count);
         coord.Y++;
      }
   }
}
extern void print_line(int line, const char* pattern, ...) {
   auto* p = (void**)&pattern;

   CONSOLE_SCREEN_BUFFER_INFO csbi;
   HANDLE hStdOut = GetStdHandle(STD_OUTPUT_HANDLE);
   if (GetConsoleScreenBufferInfo(hStdOut, &csbi)) {
      COORD coord;
      coord.X = 0;
      coord.Y = line;

      char* chars = (char*)alloca(csbi.dwSize.X + 1);
      sprintf_s(chars, csbi.dwSize.X + 1, pattern, p[1], p[2], p[3], p[4], p[5], p[6]);
      memset(chars + strlen(chars), ' ', csbi.dwSize.X - strlen(chars));

      DWORD count;
      WriteConsoleOutputCharacterA(hStdOut, chars, csbi.dwSize.X, coord, &count);
   }
}

void ins::IImage2DModel::print_image(int at_line) {
   char row[65];
   for (int i = 0; i < 32; i++) {
      for (int j = 0; j < 32; j++) {
         auto c = estimate_pixel(i, j) ? '\xb2' : '\xb0';
         row[j * 2 + 0] = c;
         row[j * 2 + 1] = c;
      }
      row[64] = 0;
      print_line(at_line + i, row);
   }
}

namespace ins {
   namespace Shapes {

      struct DenseShape {
         uint32_t height;
         uint32_t input_width;
         uint32_t output_width;
         uint32_t internal_width;
         uint32_t internal_height;
         DenseShape(uint32_t height, uint32_t input_width, uint32_t output_width, uint32_t internal_width = 0, uint32_t internal_height = 0)
            :height(height), input_width(input_width), output_width(output_width), internal_width(internal_width), internal_height(internal_height)
         {
            if (this->internal_width == 0) {
               this->internal_height = 0;
               this->internal_width = this->input_width;
            }
            else if (this->internal_height == 0) {
               this->internal_height = this->height / 2;
            }
         }
         uint32_t get_layer_width(int i) {
            if (i < internal_height) {
               float ratio = float(i) / float(internal_height);
               return input_width + uint32_t(ratio * int32_t(internal_width - input_width));
            }
            else {
               float ratio = float(i - internal_height) / float(height - 1 - internal_height);
               return internal_width + uint32_t(ratio * int32_t(output_width - internal_width));
            }
         }
      };
   }
}

struct halfspace1_image : IImage2DModel {
   bool estimate_pixel(uint8_t i, uint8_t j) override {
      return (2 * int(i) - 1 * int(j)) < 8;
   }
};

struct halfspace2_image : IImage2DModel {
   bool estimate_pixel(uint8_t i, uint8_t j) override {
      return (-2 * int(i) - 1 * int(j)) < -40;
   }
};

struct halfspace3_image : IImage2DModel {
   bool estimate_pixel(uint8_t i, uint8_t j) override {
      return (int(i) - 2 * int(j)) < 2;
   }
};

struct halfspace4_image : IImage2DModel {
   bool estimate_pixel(uint8_t i, uint8_t j) override {
      return (int(i) - 3 * int(j)) > 16;
   }
};

struct band_image : IImage2DModel {
   bool estimate_pixel(uint8_t i, uint8_t j) override {
      auto p1 = (int(i) - 2 * int(j)) < 2;
      auto p2 = (int(i) - 3 * int(j)) < 16;
      return p1 ^ p2;
   }
};

struct circles_image : IImage2DModel {
   bool estimate_pixel(uint8_t i, uint8_t j) override {
      auto r1 = (int(i) - 8) * (int(i) - 6) + (int(j) - 8) * (int(j) - 2);
      auto c1 = r1 < 9;

      auto r2 = (int(i) - 20) * (int(i) - 6) + (int(j) - 16) * (int(j) - 16);
      auto c2 = r2 < 16;

      return c1 || c2;
   }
};


int main() {

   //halfspace4_image image_ref;
   // halfspace1_image image_ref;
    //band_image image_ref;
    circles_image image_ref;

   //Models::SingleGateImage2DModel model;
   Models::HiddenLayerImage2DModel model;

   print_clean();
   print_line(3, "> dataset:");
   image_ref.print_image();
   Sleep(500);

   size_t epoch_count = 10000;
   size_t cycle_count = 100;
   double lrate = 1.0;
   for (size_t e = 0; e < epoch_count; e++) {
      for (size_t c = 0; c < cycle_count; c++) {
         size_t i = rand() % 32;
         size_t j = rand() % 32;
         auto expected = image_ref.estimate_pixel(i, j);
         model.train_pixel(i, j, expected, lrate);
         //lrate *= 0.9999;
      }
      print_line(3, "> iteration: %d (rate=%lg)", e * cycle_count, lrate);
      model.print_image();
   }

   return 0;
}
