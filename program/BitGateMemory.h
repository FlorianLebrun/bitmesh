#pragma once

#include <vector>
#include <stdint.h>

namespace ins {
   struct Page;
   struct DescriptorPage;
   struct Memory;
   struct Layer;

   typedef float Scalar;

   struct Probabilistic {

      typedef int32_t tSignal;

      struct GateStats {

         // Rx : reward when consencus at x
         tSignal R0 = 0;
         tSignal R1 = 0;

         // Px : penalty when consencus at x
         tSignal P0 = 0;
         tSignal P1 = 0;

         void add(bool gate_state, tSignal signal) {
            if (gate_state) {
               if (signal > 0) this->R1 += signal;
               else this->P1 += -signal;
            }
            else {
               if (signal > 0) this->R0 += signal;
               else this->P0 += -signal;
            }
         }
      };
      struct LinkStats {

         // Rx_Iy : reward when consencus at x and input at y
         tSignal R0_I0 = 0;
         tSignal R0_I1 = 0;
         tSignal R1_I0 = 0;
         tSignal R1_I1 = 0;

         // Px_Iy : penalty when consencus at x and input at y
         tSignal P0_I0 = 0;
         tSignal P0_I1 = 0;
         tSignal P1_I0 = 0;
         tSignal P1_I1 = 0;

         tSignal Px() {
            return this->P0_I0 + this->P0_I1 + this->P1_I0 + this->P1_I1;
         }
         tSignal Rx() {
            return this->R0_I0 + this->R0_I1 + this->R1_I0 + this->R1_I1;
         }
         void add(bool gate_state, bool link_state, tSignal signal) {
            if (gate_state) {
               if (signal > 0) {
                  if (link_state) this->R1_I1 += signal;
                  else this->R1_I0 += signal;
               }
               else {
                  if (link_state) this->P1_I1 += -signal;
                  else this->P1_I0 += -signal;
               }
            }
            else {
               if (signal > 0) {
                  if (link_state) this->R0_I1 += signal;
                  else this->R0_I0 += signal;
               }
               else {
                  if (link_state) this->P0_I1 += -signal;
                  else this->P0_I0 += -signal;
               }
            }
         }
      };
   };

   struct GateEmpty {
      struct Gate {
         struct Stats {
            void write_feedback(int32_t signal) {
               throw "";
            }
         };
         struct Parameter {
         };
      };
      struct Link {
         struct Stats {
         };
         struct Parameter {
         };
      };
   };

   struct BitPointer {
      uint32_t page_index = 0;
      uint32_t gate_index = 0;
   };

   template<class GatePolicy>
   struct BitMesh {
      typedef typename GatePolicy::Gate::Stats Gate_Stats;
      typedef typename GatePolicy::Gate::Parameter Gate_Parameter;
      typedef typename GatePolicy::Link::Stats Link_Stats;
      typedef typename GatePolicy::Link::Parameter Link_Parameter;

      union UnitStats {
         Gate_Stats gate;
         Link_Stats link;
      };

      union UnitParameter {
         Gate_Parameter gate;
         Link_Parameter link;
      };

      struct DescriptorPage {

         // Parameters infos
         UnitStats* units_stats;
         UnitParameter* units_param;
         uint32_t units_count = 0;

         // States infos
         uint32_t gates_count = 0;
         uint32_t gates_width = 0;
         uint32_t gates_bytes = 0;
      };

      struct Page {
         uint32_t page_index = 0;
         uint32_t gates_count = 0;
         uint32_t gates_width = 0;

         // Gates maps
         BitPointer* gates_links = 0; // BitLink[descriptor.units_count]
         uint8_t* gates_states = 0; // BitLink[descriptor.states_count]

         // Gates params maps
         UnitStats* units_stats = 0;
         UnitParameter* units_param = 0;

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
            this->units_stats[param_index].gate.write_feedback(feedback_signal);
         }
         BitPointer* get_gate_links(uint32_t gate_index) {
            auto param_index = get_gate_paramIdx(gate_index);
            return &this->gates_links[param_index];
         }
         Gate_Stats& get_gate_stats(uint32_t gate_index) {
            auto param_index = get_gate_paramIdx(gate_index);
            return this->units_stats[param_index].gate;
         }
         void initialize(Memory* mem) {
            GateInstance gate(mem, this, 0, 0, this->gates_width);
            for (gate.gate_index = 0; gate.gate_index < this->gates_count; gate.gate_index++) {
               GatePolicy::initialize(gate, this->gates_width);
               gate.param_index += this->gates_width;
               gate.param_end += this->gates_width;
            }
         }
         void compute_forward(Memory* mem) {
            GateInstance gate(mem, this, 0, 0, this->gates_width);
            for (gate.gate_index = 0; gate.gate_index < this->gates_count; gate.gate_index++) {
               GatePolicy::compute_forward(gate);
               gate.param_index += this->gates_width;
               gate.param_end += this->gates_width;
            }
         }
         void compute_backward(Memory* mem) {
            GateInstance gate(mem, this, 0, 0, this->gates_width);
            for (gate.gate_index = 0; gate.gate_index < this->gates_count; gate.gate_index++) {
               GatePolicy::compute_backward(gate);
               gate.param_index += this->gates_width;
               gate.param_end += this->gates_width;
            }
         }
         void mutate_forward(Memory* mem) {
            GateInstance gate(mem, this, 0, 0, this->gates_width);
            for (gate.gate_index = 0; gate.gate_index < this->gates_count; gate.gate_index++) {
               GatePolicy::mutate_forward(gate);
               gate.param_index += this->gates_width;
               gate.param_end += this->gates_width;
            }
         }
         void mutate_backward(Memory* mem) {
            GateInstance gate(mem, this, 0, 0, this->gates_width);
            for (gate.gate_index = 0; gate.gate_index < this->gates_count; gate.gate_index++) {
               GatePolicy::mutate_backward(gate);
               gate.param_index += this->gates_width;
               gate.param_end += this->gates_width;
            }
         }
      };

      struct GateInstance {
         Memory* mem;
         Page* page;
         int gate_index;
         int param_index;
         int param_end;

         struct link_iterator {
            GateInstance& gate;
            int link_index;
            link_iterator(GateInstance& gate, int32_t link_index)
               : gate(gate), link_index(link_index) {
            }
            bool operator !=(GateInstance& gate) {
               return this->link_index < gate.param_end;
            }
            void operator ++() {
               this->link_index++;
            }
            link_iterator& operator *() {
               return *this;
            }
            bool get() {
               return gate.mem->get_gate_state(gate.page->gates_links[this->link_index]);
            }
            Link_Parameter& param() {
               return gate.page->units_param[this->link_index].link;
            }
            Link_Stats& stats() {
               return gate.page->units_stats[this->link_index].link;
            }
            void emit_feeback(int32_t lfeedback) {
               gate.mem->emit_gate_feeback(gate.page->gates_links[link_index], lfeedback);
            }
         };

         GateInstance(Memory* mem, Page* page, int gate_index, int param_index, int param_end)
            : mem(mem), page(page), gate_index(gate_index), param_index(param_index), param_end(param_end) {
         }
         bool get() {
            return this->page->get_gate_state(this->gate_index);
         }
         void set(bool state) {
            this->page->set_gate_state(this->gate_index, state);
         }
         Gate_Parameter& param() {
            return this->page->units_param[this->gate_index].gate;
         }
         Gate_Stats& stats() {
            return this->page->units_stats[this->gate_index].gate;
         }
         link_iterator at(int32_t link_index) {
            return link_iterator(*this, link_index);
         }
         link_iterator begin() {
            return link_iterator(*this, this->gate_index + 1);
         }
         GateInstance& end() {
            return *this;
         }
      };

      struct Memory {
         Page* pages = 0;
         DescriptorPage* descriptors = 0;
         size_t count = 0;

         struct {
            size_t used_memory = 0;
            size_t units_count = 0;
            size_t links_count = 0;
            size_t gates_count = 0;
         } stats;

         Memory(Page* gates, DescriptorPage* descriptors, size_t count)
            :pages(gates), descriptors(descriptors), count(count) {
         }

         bool get_gate_state(BitPointer ptr) {
            auto& page = this->pages[ptr.page_index];
            return page.get_gate_state(ptr.gate_index);
         }
         BitPointer* get_gate_links(BitPointer ptr) {
            auto& page = this->pages[ptr.page_index];
            return page.get_gate_links(ptr.gate_index);
         }
         Gate_Stats& get_gate_stats(BitPointer ptr) {
            auto& page = this->pages[ptr.page_index];
            return page.get_gate_stats(ptr.gate_index);
         }
         void initialize() {
            for (intptr_t i = 0; i < count; i++) {
               auto& page = this->pages[i];
               if (page.gates_links) page.initialize(this);
            }
         }
         void compute_forward() {
            for (intptr_t i = 0; i < count; i++) {
               auto& page = this->pages[i];
               if (page.gates_links) page.compute_forward(this);
            }
         }
         void compute_backward() {
            for (intptr_t i = count - 1; i >= 0; i--) {
               auto& page = this->pages[i];
               if (page.gates_links) page.compute_backward(this);
            }
         }
         void mutate_forward() {
            for (intptr_t i = 0; i < count; i++) {
               auto& page = this->pages[i];
               if (page.gates_links) page.mutate_forward(this);
            }
         }
         void mutate_backward() {
            for (intptr_t i = count - 1; i >= 0; i--) {
               auto& page = this->pages[i];
               if (page.gates_links) page.mutate_backward(this);
            }
         }
         void* create_buffer(size_t element_count, size_t element_size) {
            this->stats.used_memory += element_size * element_count;
            auto buf = malloc(element_size * element_count);
            memset(buf, 0, element_size * element_count);
            return buf;
         }
         void create_descriptor(uint32_t desc_index, uint32_t gates_count, uint32_t gates_width) {
            auto& desc = this->descriptors[desc_index];

            // Set gates page sizing
            desc.units_count = gates_width * gates_count;
            desc.gates_width = gates_width;
            desc.gates_count = gates_count;
            desc.gates_bytes = desc.gates_count / 8;
            if (desc.gates_bytes * 8 < desc.gates_count)desc.gates_bytes++;
            this->stats.units_count += desc.units_count;

            // Allocate params memory
            desc.units_stats = (UnitStats*)this->create_buffer(desc.units_count, sizeof(UnitStats));
            desc.units_param = (UnitParameter*)this->create_buffer(desc.units_count, sizeof(UnitParameter));
         }
         void create_page(uint32_t page_index, DescriptorPage* descriptor) {
            auto& page = this->pages[page_index];
            page.page_index = page_index;
            page.units_stats = descriptor->units_stats;
            page.units_param = descriptor->units_param;
            page.gates_count = descriptor->gates_count;
            page.gates_width = descriptor->gates_width;
            if (page.gates_width > 1) {
               page.gates_links = (BitPointer*)this->create_buffer(descriptor->units_count, sizeof(BitPointer*));
            }
            else {
               page.gates_links = 0;
            }
            page.gates_states = (uint8_t*)this->create_buffer(descriptor->gates_bytes, sizeof(uint8_t));

            this->stats.gates_count += page.gates_count;
            this->stats.links_count += page.gates_count * (page.gates_width - 1);
         }

         void emit_gate_feeback(BitPointer ptr, Scalar signal) {
            auto& page = this->pages[ptr.page_index];
            return page.emit_gate_feeback(ptr.gate_index, signal);
         }
         void emit_layer_feeback(Layer& layer, std::vector<Scalar> signals) {
            if (layer.shape.ndims != 1) throw;
            if (layer.shape.dims[0] != signals.size()) throw;

            BitPointer ptr;
            int index = 0;
            for (ptr.page_index = layer.layout.page_index_first; ptr.page_index <= layer.layout.page_index_last; ptr.page_index++) {
               auto gates_count = this->pages[ptr.page_index].gates_count;
               for (ptr.gate_index = 0; ptr.gate_index < gates_count; ptr.gate_index++) {
                  this->emit_gate_feeback(ptr, signals[index]);
                  index++;
               }
            }
         }

      };

   };
}
