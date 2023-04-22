#pragma once

#include <vector>
#include <stdint.h>

namespace ins {
   struct BitGatePage;
   struct BitDescriptorPage;
   struct BitGateMemory;

   struct BitGateStats {
      // R_Cx : reward when consencus at x
      uint32_t R_C0 = 0;
      uint32_t R_C1 = 0;
      // P_Cx : penalty when consencus at x
      uint32_t P_C0 = 0;
      uint32_t P_C1 = 0;
      // Links weights sum
      int32_t weights_sum = 0;
      // Accumulated feedback
      int32_t feedback_signal = 0;
   };

   struct BitLinkStats {
      // R_Cx_Iy : reward when consencus at x and input at y
      uint32_t R_C0_I0 = 0;
      uint32_t R_C0_I1 = 0;
      uint32_t R_C1_I0 = 0;
      uint32_t R_C1_I1 = 0;
      // P_Cx_Iy : penalty when consencus at x and input at y
      uint32_t P_C0_I0 = 0;
      uint32_t P_C0_I1 = 0;
      uint32_t P_C1_I0 = 0;
      uint32_t P_C1_I1 = 0;
      // Paremeters
      uint32_t drift = 0;          // mutation asymetry factor (~ around 1.0)
   };

   union BitParameterStats {
      BitGateStats gate_stats;
      BitLinkStats link_stats;
   };

   typedef int16_t BitParameterValue;

   struct BitPointer {
      uint32_t page_index = 0;
      uint32_t gate_index = 0;
   };

   struct BitDescriptorPage {

      // Parameters infos
      BitParameterStats* params_stats;
      BitParameterValue* params_values;
      uint32_t params_count = 0;

      // States infos
      uint32_t gates_count = 0;
      uint32_t gates_width = 0;
      uint32_t gates_bytes = 0;
   };

   struct BitGatePage {
      uint32_t page_index = 0;
      uint32_t gates_count = 0;
      uint32_t gates_width = 0;

      // Gates maps
      BitPointer* gates_links = 0; // BitLink[descriptor.params_count]
      uint8_t* gates_states = 0; // BitLink[descriptor.states_count]

      // Gates params maps
      BitParameterStats* params_stats = 0;
      BitParameterValue* params_values = 0;

      uint32_t get_gate_paramIdx(uint32_t gate_index) {
         return this->gates_width * gate_index;
      }
      bool get_gate_state(uint32_t gate_index) {
         uint8_t& bit_state = this->gates_states[gate_index / 8];
         uint8_t bit_mask = 1 << (gate_index & 7);
         if (bit_state & bit_mask) return true;
         return false;
      }
      void set_gate_state(uint32_t gate_index, bool value) {
         uint8_t& bit_state = this->gates_states[gate_index / 8];
         uint8_t bit_mask = 1 << (gate_index & 7);
         if (value) bit_state |= bit_mask;
         else bit_state &= ~bit_mask;
      }
      void emit_gate_feeback(uint32_t gate_index, int32_t feedback_signal) {
         uint8_t& bit_state = this->gates_states[gate_index / 8];
         uint8_t bit_mask = 1 << (gate_index & 7);
         auto param_index = get_gate_paramIdx(gate_index);
         this->params_stats[param_index].gate_stats.feedback_signal += feedback_signal;
      }
      BitPointer* get_gate_links(uint32_t gate_index) {
         auto param_index = get_gate_paramIdx(gate_index);
         return &this->gates_links[param_index];
      }
      BitGateStats& get_gate_stats(uint32_t gate_index) {
         auto param_index = get_gate_paramIdx(gate_index);
         return this->params_stats[param_index].gate_stats;
      }
      template<class GatePolicy>
      void compute_forward(BitGateMemory* memory) {
         uint32_t param_index = 0;
         for (int gate_index = 0; gate_index < this->gates_count; gate_index++) {
            GatePolicy::compute_forward(memory, this, gate_index, param_index);
            param_index += this->gates_width;
         }
      }
      template<class GatePolicy>
      void compute_backward(BitGateMemory* mem) {
         uint32_t param_index = 0;
         for (int gate_index = 0; gate_index < this->gates_count; gate_index++) {
            GatePolicy::compute_backward(mem, this, gate_index, param_index);
            param_index += this->gates_width;
         }
      }
      template<class GatePolicy>
      void mutate_forward(BitGateMemory* mem) {
         uint32_t param_index = 0;
         for (int gate_index = 0; gate_index < this->gates_count; gate_index++) {
            GatePolicy::mutate_forward(mem, this, gate_index, param_index);
            param_index += this->gates_width;
         }
      }
      template<class GatePolicy>
      void mutate_backward(BitGateMemory* mem) {
         uint32_t param_index = 0;
         for (int gate_index = 0; gate_index < this->gates_count; gate_index++) {
            GatePolicy::mutate_backward(mem, this, gate_index, param_index);
            param_index += this->gates_width;
         }
      }
   };

   struct BitGateMemory {
      BitGatePage* pages = 0;
      BitDescriptorPage* descriptors = 0;
      size_t count = 0;

      struct {
         size_t used_memory = 0;
         size_t params_count = 0;
         size_t links_count = 0;
         size_t gates_count = 0;
      } stats;

      BitGateMemory(BitGatePage* gates, BitDescriptorPage* descriptors, size_t count)
         :pages(gates), descriptors(descriptors), count(count) {
      }

      void create_descriptor(uint32_t desc_index, uint32_t gates_count, uint32_t gates_width);
      void create_page(uint32_t page_index, BitDescriptorPage* descriptor);
      void* create_buffer(size_t element_count, size_t element_size);

      BitPointer* get_gate_links(BitPointer ptr);
      BitGateStats& get_gate_stats(BitPointer ptr);
      bool get_gate_state(BitPointer ptr);
      void emit_gate_feeback(BitPointer ptr, int32_t signal);

      template<class GatePolicy>
      void compute_forward();
      template<class GatePolicy>
      void compute_backward();
      template<class GatePolicy>
      void mutate_forward();
      template<class GatePolicy>
      void mutate_backward();
   };

   bool BitGateMemory::get_gate_state(BitPointer ptr) {
      auto& page = this->pages[ptr.page_index];
      return page.get_gate_state(ptr.gate_index);
   }
   void BitGateMemory::emit_gate_feeback(BitPointer ptr, int32_t signal) {
      auto& page = this->pages[ptr.page_index];
      return page.emit_gate_feeback(ptr.gate_index, signal);
   }
   BitPointer* BitGateMemory::get_gate_links(BitPointer ptr) {
      auto& page = this->pages[ptr.page_index];
      return page.get_gate_links(ptr.gate_index);
   }
   BitGateStats& BitGateMemory::get_gate_stats(BitPointer ptr) {
      auto& page = this->pages[ptr.page_index];
      return page.get_gate_stats(ptr.gate_index);
   }
   template<class GatePolicy>
   void BitGateMemory::compute_forward() {
      for (intptr_t i = 0; i < count; i++) {
         auto& page = this->pages[i];
         if (page.gates_links) page.compute_forward<GatePolicy>(this);
      }
   }
   template<class GatePolicy>
   void BitGateMemory::compute_backward() {
      for (intptr_t i = count - 1; i >= 0; i--) {
         auto& page = this->pages[i];
         if (page.gates_links) page.compute_backward<GatePolicy>(this);
      }
   }
   template<class GatePolicy>
   void BitGateMemory::mutate_forward() {
      for (intptr_t i = 0; i < count; i++) {
         auto& page = this->pages[i];
         if (page.gates_links) page.mutate_forward<GatePolicy>(this);
      }
   }
   template<class GatePolicy>
   void BitGateMemory::mutate_backward() {
      for (intptr_t i = count - 1; i >= 0; i--) {
         auto& page = this->pages[i];
         if (page.gates_links) page.mutate_backward<GatePolicy>(this);
      }
   }
   void* BitGateMemory::create_buffer(size_t element_count, size_t element_size) {
      this->stats.used_memory += element_size * element_count;
      return malloc(element_size * element_count);
   }
   void BitGateMemory::create_descriptor(uint32_t desc_index, uint32_t gates_count, uint32_t gates_width) {
      auto& desc = this->descriptors[desc_index];

      // Set gates page sizing
      desc.params_count = gates_width * gates_count;
      desc.gates_width = gates_width;
      desc.gates_count = gates_count;
      desc.gates_bytes = desc.gates_count / 8;
      if (desc.gates_bytes * 8 < desc.gates_count)desc.gates_bytes++;
      this->stats.params_count += desc.params_count;

      // Allocate params memory
      desc.params_stats = (BitParameterStats*)this->create_buffer(desc.params_count, sizeof(BitParameterStats));
      desc.params_values = (BitParameterValue*)this->create_buffer(desc.params_count, sizeof(BitParameterValue));

      // Define weights distribution stats
      uint16_t gate_scaleL2 = 8;
      uint32_t gate_weight_range = gates_width * (uint32_t(1) << gate_scaleL2) / desc.gates_width;
      uint32_t gate_weight_offset = 0.5 * gate_weight_range;

      // Initiate gate parameters
      uint32_t param_index = 0;
      for (int gate_index = 0; gate_index < desc.gates_count; gate_index++) {
         auto param_end = param_index + desc.gates_width;

         // Initiate gate core param
         auto& stats = desc.params_stats[param_index].gate_stats;
         desc.params_values[param_index] = gate_scaleL2;
         stats.BitGateStats::BitGateStats();
         param_index++;

         // Initiate gate links param
         uint32_t weights_sum = 0;
         while (param_index < param_end) {
            auto& lstats = desc.params_stats[param_index].link_stats;
            auto& lweight = desc.params_values[param_index];
            lstats.BitLinkStats::BitLinkStats();

            lweight = (rand() % gate_weight_range) - gate_weight_offset;
            weights_sum += std::abs(lweight);

            param_index++;
         }

         stats.weights_sum = weights_sum;
      }
   }
   void BitGateMemory::create_page(uint32_t page_index, BitDescriptorPage* descriptor) {
      auto& page = this->pages[page_index];
      page.page_index = page_index;
      page.params_stats = descriptor->params_stats;
      page.params_values = descriptor->params_values;
      page.gates_count = descriptor->gates_count;
      page.gates_width = descriptor->gates_width;
      if (page.gates_width > 1) {
         page.gates_links = (BitPointer*)this->create_buffer(descriptor->params_count, sizeof(BitPointer*));
      }
      else {
         page.gates_links = 0;
      }
      page.gates_states = (uint8_t*)this->create_buffer(descriptor->gates_bytes, sizeof(uint8_t));
      memset(page.gates_states, 0, descriptor->gates_bytes);

      this->stats.gates_count += page.gates_count;
      this->stats.links_count += page.gates_count * (page.gates_width - 1);
   }
}
