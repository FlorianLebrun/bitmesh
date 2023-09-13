#include "./BitGateMemory.h"
#include "./BitGateModel.h"
#include "./BitGate_basic_policy.h"
#include "./BitGate_stats_policy.h"
#include "./BitGate_shadow_policy.h"
#include "./BitGate_randmut_policy.h"
#include <stdio.h>
#include <windows.h>

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
using namespace ins;


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

void layer_write_vec8(Layer& layer, std::vector<uint8_t> values, BitGateMemory mem) {
   if (values.size() != layer.shape.dims[0]) throw;
   if (layer.shape.dims[1] != 8) throw;
   if (layer.shape.ndims != 2) throw;

   int pos = 0;
   for (size_t i = layer.layout.page_index_first; i < layer.layout.page_index_last; i++) {
      auto count = layer.layout.gate_per_base_page / 8;
      memcpy(mem->pages[i].gates_states, &values[pos], count);
      pos += count;
   }
   memcpy(mem->pages[layer.layout.page_index_last].gates_states, &values[pos], layer.layout.gate_per_last_page / 8);
}

std::vector<bool> layer_read_vec1(Layer& layer, BitGateMemory mem) {
   if (layer.shape.ndims != 1) throw;
   std::vector<bool> vec;
   layer.foreach_gate(mem,
      [&](BitPointer gate) {
         auto r = mem->get_gate_state(gate);
         if (r) vec.push_back(true);
         else vec.push_back(false);
      }
   );
   return std::forward<std::vector<bool>>(vec);
}

void print_layers_stats(Model& model, BitGateMemory mem) {
   printf("____________________________________________\n");
   for (auto layer : model.layers) {
      //printf("%.3d: %.3dpages | ", layer->id, layer->layout.base_page_count + 1);
      layer->foreach_gate(mem,
         [&](BitPointer gate) {
            auto& stats = mem->get_gate_stats(gate);
            if (mem->get_gate_state(gate)) printf("\xb2");
            else printf("\xb0");
         }
      );
      printf("\n");
   }
}

void print_layers_states(Model& model, BitGateMemory mem) {
   printf("____________________________________________\n");
   for (auto layer : model.layers) {
      //printf("%.3d: %.3dpages | ", layer->id, layer->layout.base_page_count + 1);
      layer->foreach_gate(mem,
         [&](BitPointer gate) {
            if (mem->get_gate_state(gate)) printf("\xb2");
            else printf("\xb0");
         }
      );
      printf("\n");
   }
}

void print_image(int line, std::function<bool(uint8_t, uint8_t)>&& eval) {
   char row[65];
   for (int i = 0; i < 32; i++) {
      for (int j = 0; j < 32; j++) {
         auto c = eval(i, j) ? '\xb2' : '\xb0';
         row[j * 2 + 0] = c;
         row[j * 2 + 1] = c;
      }
      row[64] = 0;
      print_line(line + i, row);
   }
}

bool halfspace1_image(uint8_t i, uint8_t j) {
   return (2 * int(i) - 1 * int(j)) < 8;
}

bool halfspace2_image(uint8_t i, uint8_t j) {
   return (-2 * int(i) - 1 * int(j)) < -40;
}

bool band_image(uint8_t i, uint8_t j) {
   auto p1 = (int(i) - 2 * int(j)) < 2;
   auto p2 = (int(i) - 3 * int(j)) < 16;
   return p1 ^ p2;
}

bool circles_image(uint8_t i, uint8_t j) {
   auto r1 = (int(i) - 8) * (int(i) - 6) + (int(j) - 8) * (int(j) - 2);
   auto c1 = r1 < 9;

   auto r2 = (int(i) - 20) * (int(i) - 6) + (int(j) - 16) * (int(j) - 16);
   auto c2 = r2 < 16;

   return c1 || c2;
}

int main() {
   FullConnectBinder full_binder;

   Model model;
#if 0
   auto& lay_in = model.lay(Shape(2, 8));
   auto& lay_1 = model.lay(Shape(2)).on(lay_in, &full_binder);
   auto& lay_out = model.lay(Shape(1)).on(lay_1, &full_binder);
#else
   auto& lay_in = model.lay(Shape(2, 8));
   auto& lay_out = model.lay(Shape(1)).on(lay_in, &full_binder);
#endif
   auto mem = model.materialize<RandMutationGatePolicy>();

   auto estimate_pixel = [&](uint8_t i, uint8_t j)->bool {
      layer_write_vec8(lay_in, { i, j }, mem);
      mem->compute_forward();
      auto r = layer_read_vec1(lay_out, mem);
      return r[0];
   };

   auto train_pixel = [&](uint8_t i, uint8_t j, bool expected)->bool {
      layer_write_vec8(lay_in, { i, j }, mem);
      mem->compute_forward();
      auto r = layer_read_vec1(lay_out, mem);
      if (r[0] == expected) return r[0];
       
      mem->emit_layer_feeback(lay_out, { (r[0] == expected) ? 1.0f : -1.0f });
      mem->compute_backward();
      mem->mutate_forward();
      mem->mutate_backward();
      return r[0];
   };
#define image_func halfspace1_image
//#define image_func band_image

   print_layers_states(model, mem);
   estimate_pixel(1, 6);
   print_layers_states(model, mem);

   train_pixel(1, 6, image_func(1, 6));
   print_layers_states(model, mem);

   print_clean();

   print_line(3, "> dataset:");
   print_image(4, image_func);
   Sleep(500);

   print_line(3, "> estimated:");
   size_t epoch_count = 10000;
   size_t cycle_count = 10000;
   for (size_t e = 0; e < epoch_count; e++) {
      for (size_t c = 0; c < cycle_count; c++) {
         size_t i = rand() % 32;
         size_t j = rand() % 32;
         auto expected = image_func(i, j);
         train_pixel(i, j, expected);
      }
      print_image(4, estimate_pixel);
   }

   return 0;
}
