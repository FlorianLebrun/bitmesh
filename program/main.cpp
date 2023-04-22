#include "./BitGateMemory.h"
#include "./BitGateModel.h"
#include "./BitGatePolicy.h"

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
         /*void shape(Compoment& self) {
            for (int i = 0; i < height; i++) {
               uint32_t width_max = this->get_layer_width(i);
               self.layers.push_back(Layer(width_max));

               auto& layer = self.layers.back();
               while (layer.gates.size() < width_max) {
                  this->addGate(self, i);
               }
            }
         }*/
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

void layer_write_vec8(Layer& layer, std::vector<uint8_t> values, BitGateMemory* mem) {
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

std::vector<bool> layer_read_vec1(Layer& layer, BitGateMemory* mem) {
   if (layer.shape.ndims != 1) throw;
   std::vector<bool> vec;
   layer.foreach_gate(*mem,
      [&](BitPointer gate) {
         auto r = mem->get_gate_state(gate);
         if (r) vec.push_back(true);
         else vec.push_back(false);
      }
   );
   return std::forward<std::vector<bool>>(vec);
}

void layer_emit_feeback_vec(Layer& layer, std::vector<int32_t> feedbacks, BitGateMemory* mem) {
   if (layer.shape.ndims != 1) throw;
   if (layer.shape.dims[0] != feedbacks.size()) throw;
   int index = 0;
   layer.foreach_gate(*mem,
      [&](BitPointer ptr) {
         auto r = mem->get_gate_state(ptr);
         mem->emit_gate_feeback(ptr, feedbacks[index]);
      }
   );
}

void print_layers_stats(Model& model, BitGateMemory* mem) {
   printf("____________________________________________\n");
   for (auto layer : model.layers) {
      //printf("%.3d: %.3dpages | ", layer->id, layer->layout.base_page_count + 1);
      layer->foreach_gate(*mem,
         [&](BitPointer gate) {
            auto& stats = mem->get_gate_stats(gate);
            if (mem->get_gate_state(gate)) printf("\xb2");
            else printf("\xb0");
         }
      );
      printf("\n");
   }
}

void print_layers_states(Model& model, BitGateMemory* mem) {
   printf("____________________________________________\n");
   for (auto layer : model.layers) {
      //printf("%.3d: %.3dpages | ", layer->id, layer->layout.base_page_count + 1);
      layer->foreach_gate(*mem,
         [&](BitPointer gate) {
            if (mem->get_gate_state(gate)) printf("\xb2");
            else printf("\xb0");
         }
      );
      printf("\n");
   }
}

void print_image(std::function<bool(uint8_t, uint8_t)>&& eval) {
   for (int i = 0; i < 32; i++) {
      for (int j = 0; j < 32; j++) {
         auto r = eval(i, j);
         if (r) printf("\xb2\xb2");
         else printf("\xb0\xb0");
      }
      printf("\n");
   }
}

void unwind_lines(int count) {
   while (--count) printf("\033[A");
}

bool halfspace_image(uint8_t i, uint8_t j) {
   return (2 * int(i) - 1 * int(j)) < 8;
};

bool band_image(uint8_t i, uint8_t j) {
   auto p1 = (int(i) - 2 * int(j)) < 2;
   auto p2 = (int(i) - 3 * int(j)) < 16;
   return p1 ^ p2;
};

bool circles_image(uint8_t i, uint8_t j) {
   auto r1 = (int(i) - 8) * (int(i) - 6) + (int(j) - 8) * (int(j) - 2);
   auto c1 = r1 < 9;

   auto r2 = (int(i) - 20) * (int(i) - 6) + (int(j) - 16) * (int(j) - 16);
   auto c2 = r2 < 16;

   return c1 || c2;
};

int main() {
   FullConnectBinder full_binder;

   Model model;
   auto& lay_in = model.lay(Shape(2, 8));
   auto& lay_1 = model.lay(Shape(20)).on(lay_in, &full_binder);
   auto& lay_2 = model.lay(Shape(20)).on(lay_1, &full_binder);
   auto& lay_out = model.lay(Shape(1)).on(lay_2, &full_binder);

   BitGateMemory* mem = model.materialize();

   auto estimate_pixel = [&](uint8_t i, uint8_t j)->bool {
      layer_write_vec8(lay_in, { i, j }, mem);
      mem->compute_forward<BitGatePolicy>();
      auto r = layer_read_vec1(lay_out, mem);
      return r[0];
   };

   auto train_pixel = [&](uint8_t i, uint8_t j, bool expected)->bool {
      layer_write_vec8(lay_in, { i, j }, mem);
      mem->compute_forward<BitGatePolicy>();
      auto r = layer_read_vec1(lay_out, mem);
      layer_emit_feeback_vec(lay_out, { r[0] == expected ? 10000 : -10000 }, mem);
      mem->compute_backward<BitGatePolicy>();
      mem->mutate_forward<BitGatePolicy>();
      mem->mutate_backward<BitGatePolicy>();
      return r[0];
   };

   print_layers_states(model, mem);
   estimate_pixel(1, 6);
   print_layers_states(model, mem);

   train_pixel(1, 6, halfspace_image(1, 6));
   print_layers_states(model, mem);

   size_t epoch_count = 100;
   size_t cycle_count = 100;
   for (size_t e = 0; e < epoch_count; e++) {
      //print_image(estimate_pixel);
      //unwind_lines(33);
      for (size_t c = 0; c < cycle_count; c++) {
         size_t i = rand() % 32;
         size_t j = rand() % 32;
         auto expected = halfspace_image(i, j);
         train_pixel(i, j, expected);
      }
   }
   print_image(estimate_pixel);

   /*while (1)
   {
      print_image(halfspace_image);
      unwind_lines(33);
      print_image(band_image);
      unwind_lines(33);
      print_image(circles_image);
      unwind_lines(33);
   }*/
   return 0;
}
