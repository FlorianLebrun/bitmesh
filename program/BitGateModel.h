#pragma once

#include "./BitGateMemory.h"
#include <functional>
#include <thread>
#include <string>
#include <stdio.h>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <algorithm>

namespace ins {
   struct Layer;
   struct LayerBinder;
   struct LayerSupport;
   struct Model;
   struct Shape;

   typedef uint32_t LayerID;

   struct Shape {
      uint32_t dims[4] = { 0 };
      uint32_t ndims = 0;
      uint32_t width = 0;
      Shape() {}
      Shape(uint32_t d0) : dims{ d0 }, ndims(1), width(d0) {}
      Shape(uint32_t d0, uint32_t d1) : dims{ d0,d1 }, ndims(2), width(d0* d1) {}
      Shape(uint32_t d0, uint32_t d1, uint32_t d2) : dims{ d0,d1,d2 }, ndims(2), width(d0* d1* d2) {}
   };

   struct LayerBinder {
      virtual size_t get_gates_links(Layer& layer, LayerSupport& support) = 0;
      virtual void make_gates_links(Layer& layer, LayerSupport& support, BitGateMemory& mem) = 0;
   };

   struct LayerSupport {
      LayerBinder& binder;
      Layer& layer;
      size_t links_first = 0;
      size_t links_last = 0;
      LayerSupport(LayerBinder& binder, Layer& layer)
         : binder(binder), layer(layer) {
      }
   };

   struct LayerMemoryLayout {

      struct Config {
         size_t nominal_page_params_count;
         size_t min_page_gates_count;
         size_t max_page_gates_count;
      };

      // Sizing
      size_t params_per_gate = 0;
      size_t gate_count = 0;

      // Allocation
      size_t gate_per_base_page = 0;
      size_t gate_per_last_page = 0;
      size_t base_page_count = 0;
      size_t page_index_first = 0;
      size_t page_index_last = 0;

      void setup(size_t params_per_gate, size_t gate_count, size_t page_index, Config& config);
      void allocate(BitGateMemory* mem);
   };

   struct Layer {
      Shape shape;
      std::vector<LayerSupport> supports;
      Model* model = 0;
      LayerMemoryLayout layout;
      LayerID id = 0;
      Layer(LayerID id, Shape shape, Model* model)
         : id(id), shape(shape), model(model) {
      }
      Layer& on(Layer& layer, LayerBinder* binder) {
         LayerSupport support(*binder, layer);
         this->supports.push_back(support);
         return *this;
      }
      void foreach_gate(BitGateMemory& mem, std::function<void(BitPointer)>&& visitor) {
         BitPointer ptr;
         for (ptr.page_index = this->layout.page_index_first; ptr.page_index <= this->layout.page_index_last; ptr.page_index++) {
            auto gates_count = mem.pages[ptr.page_index].gates_count;
            for (ptr.gate_index = 0; ptr.gate_index < gates_count; ptr.gate_index++) {
               visitor(ptr);
            }
         }
      }

   };

   struct ModelMemoryLayout {
      size_t page_count = 0;
      size_t params_count = 0;
   };

   struct Model {
      std::vector<Layer*> layers;
      ModelMemoryLayout layout;

      Layer& lay(Shape shape);
      BitGateMemory* materialize();

      void write_vec8(std::vector<uint8_t>& values, Layer& target);
      void read_vec8(std::vector<uint8_t>& values, Layer& target);

      void write_vec1(std::vector<bool>& values, Layer& target);
      void read_vec1(std::vector<bool>& values, Layer& target);
   };

   struct FullConnectBinder : LayerBinder {
      size_t get_gates_links(Layer& layer, LayerSupport& support) override {
         return support.layer.shape.width;
      }
      void make_gates_links(Layer& layer, LayerSupport& support, BitGateMemory& mem) override {
         layer.foreach_gate(mem,
            [&](BitPointer gate) {
               auto* links = mem.get_gate_links(gate);
               auto link_index = support.links_first;
               support.layer.foreach_gate(mem,
                  [&](BitPointer link_gate) {
                     _ASSERT(link_index <= support.links_last);
                     links[link_index] = link_gate;
                     link_index++;
                  }
               );
            }
         );
      }
   };

   template <class _Ty>
   constexpr const _Ty& clamp(const _Ty& x, const _Ty& _min, const _Ty& _max) {
      return x < _min ? _min : (x > _max ? _max : x);
   }

   Layer& Model::lay(Shape shape) {
      auto layer = new Layer(this->layers.size(), shape, this);
      this->layers.push_back(layer);
      return *layer;
   }

   void LayerMemoryLayout::setup(size_t params_per_gate, size_t gate_count, size_t page_index, Config& config) {
      this->params_per_gate = params_per_gate;
      this->gate_count = gate_count;

      // Compute base page allocation
      this->gate_per_base_page = ins::clamp(
         config.nominal_page_params_count / params_per_gate,
         config.min_page_gates_count,
         config.max_page_gates_count
      );
      if (this->gate_per_base_page > gate_count) {
         this->gate_per_base_page = gate_count;
      }
      this->base_page_count = gate_count / this->gate_per_base_page;

      // Compute last page allocation
      if (this->base_page_count * this->gate_per_base_page < gate_count) {
         this->gate_per_last_page = gate_count - this->base_page_count * this->gate_per_base_page;
      }
      else {
         this->gate_per_last_page = this->gate_per_base_page;
         this->base_page_count--;
      }

      // Set page location
      this->page_index_first = page_index;
      this->page_index_last = page_index + this->base_page_count;
   }

   void LayerMemoryLayout::allocate(BitGateMemory* mem) {
      auto allocate_page = [&](size_t i, size_t gates_count) {
         auto& desc = mem->descriptors[i];
         mem->create_descriptor(i, gates_count, this->params_per_gate);
         mem->create_page(i, &desc);
      };
      for (size_t i = this->page_index_first; i < this->page_index_last; i++) {
         allocate_page(i, this->gate_per_base_page);
      }
      allocate_page(this->page_index_last, this->gate_per_last_page);
   }

   BitGateMemory* Model::materialize() {

      LayerMemoryLayout::Config config;
      config.nominal_page_params_count = 1024;
      config.min_page_gates_count = 8;
      config.max_page_gates_count = 64;

      // Compute layers memory layout
      size_t page_count = 0;
      size_t params_count = 0;
      for (auto layer : this->layers) {
         size_t gate_links_count = 0;
         for (auto& sup : layer->supports) {
            auto sup_count = sup.binder.get_gates_links(*layer, sup);
            sup.links_first = gate_links_count;
            sup.links_last = gate_links_count + sup_count - 1;
            gate_links_count += sup_count;
         }
         layer->layout.setup(gate_links_count + 1, layer->shape.width, page_count, config);
         page_count = layer->layout.page_index_last + 1;
         params_count += layer->layout.params_per_gate * layer->layout.gate_count;
      }
      this->layout.page_count = page_count;
      this->layout.params_count = params_count;

      // Instanciate layers memory
      auto gates = new BitGatePage[page_count];
      auto descriptors = new BitDescriptorPage[page_count];
      auto mem = new BitGateMemory(gates, descriptors, page_count);
      for (auto layer : this->layers) {
         layer->layout.allocate(mem);
         if (layer->supports.size()) {
            for (auto& sup : layer->supports) {
               sup.binder.make_gates_links(*layer, sup, *mem);
            }
            layer->foreach_gate(*mem,
               [&](BitPointer gate) {
                  mem->get_gate_links(gate)[layer->layout.params_per_gate - 1] = gate;
               }
            );
         }
      }

      return mem;
   }

}
