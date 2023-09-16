#pragma once

#include "../math.h"
#include <functional>
#include <memory>

std::random_device generator;
std::uniform_real_distribution<double> distribution_unsigned(0.0, 1.0);
std::uniform_real_distribution<double> distribution_signed(-1.0, 1.0);

namespace ins {
   struct GateObject {

      typedef int32_t weight_t;
      typedef int64_t weight_sum_t;

      static constexpr weight_t WeightMax = 10000;
      static constexpr weight_t WeightMin = -WeightMax;

      struct Gate {
         weight_t weight_base = 0;

         Scalar mut_prob_neg = 0;
         Scalar mut_prob_pos = 0;

         Scalar mutation_signal = 0;
         Scalar mutation_inhibit = 0;
      };

      struct Link {
         GateObject& source;

         weight_t weight = 0;

         Scalar mut_prob_neg = 0;
         Scalar mut_prob_pos = 0;

         Link(GateObject& source)
            : source(source) {
         }
      };

      bool state = 0;

      Gate gate;
      std::vector<Link> links;

      void emit_feeback(Scalar mutation_signal) {
         if (links.size() == 0) return;

         gate.mutation_signal += mutation_signal;
      }
      void initialize() {
#if 1
         for (auto& link : links) {
            link.weight = random_signed() * 1000;
         }
         gate.weight_base = random_signed() * 1000;
#else
         for (auto& link : links) {
            link.weight = 0;
         }
         gate.weight_base = 0;
#endif
      }

      void compute_forward() {
         if (links.size() == 0) return;

         // Flush integrated signal
         auto mutation_inhibit = gate.mutation_inhibit;
         gate.mutation_inhibit = 0;
         if (mutation_inhibit) {

         }

         // Compute value
         weight_sum_t acc = gate.weight_base;
         for (auto& link : links) {
            if (link.source.state) acc += link.weight;
         }
         this->state = (acc > 0);
      }

      void compute_backward() {
         if (links.size() == 0) return;

         // Flush integrated signal
         auto mutation_signal = gate.mutation_signal;
         gate.mutation_signal = 0;
         auto mutation_inhibit = gate.mutation_inhibit;
         gate.mutation_inhibit = 0;
         if (mutation_inhibit) {

         }

         // Compute links weights sum
         weight_sum_t links_weights_sum = gate.weight_base;
         for (auto& link : links) {
            if (link.source.state) links_weights_sum += abs(link.weight);
         }

         // Compute mutation signal distribution params
         Scalar feedback_prob = 1.0 * std::abs(mutation_signal);
         Scalar feedback_factor = links_weights_sum > 0 ? (0.99 / Scalar(links_weights_sum)) : 0;
         Scalar feedback_offset = (1.0 - feedback_factor * links_weights_sum) / Scalar(links.size());
         Scalar reward_damping = 0.0;

         // Integrate mutation signal to stats
         //--- integrate to gate stats
         {
            if (mutation_signal > 0) {
               gate.mut_prob_neg -= feedback_prob * reward_damping;
               gate.mut_prob_pos -= feedback_prob * reward_damping;
            }
            else {
               if (this->state == 0) {
                  gate.mut_prob_pos += feedback_prob;
               }
               else {
                  gate.mut_prob_neg += feedback_prob;
               }
            }
         }
         //--- integrate to links stats
         Scalar links_feedback = 0;
         for (auto& link : links) {

            // Compute link mutation signal
            Scalar lfeedback = 0;
            if (link.source.state == 1) {
               if (mutation_signal > 0) {
                  gate.mut_prob_neg -= feedback_prob * reward_damping;
                  gate.mut_prob_pos -= feedback_prob * reward_damping;
                  lfeedback = feedback_offset + mutation_signal * abs(link.weight) * feedback_factor;
               }
               else {
                  lfeedback = feedback_offset + mutation_signal * link.weight * feedback_factor;
                  if (this->state == 0) {
                     link.mut_prob_pos += feedback_prob;
                  }
                  else {
                     link.mut_prob_neg += feedback_prob;
                  }
               }
            }
            else {
               if (mutation_signal > 0) {
                  link.mut_prob_neg *= reward_damping;
                  link.mut_prob_pos *= reward_damping;
                  lfeedback = feedback_offset + mutation_signal * abs(link.weight) * feedback_factor;
               }
               else {
                  lfeedback = -feedback_offset + mutation_signal * link.weight * feedback_factor;
                  if (this->state == 0) {
                     link.mut_prob_neg += feedback_prob;
                  }
                  else {
                     link.mut_prob_pos += feedback_prob;
                  }
               }
            }
            link.mut_prob_pos = clamp<Scalar>(link.mut_prob_pos, 0, 1);
            link.mut_prob_neg = clamp<Scalar>(link.mut_prob_neg, 0, 1);

            // Dispatch mutation signal to link input stats
            link.source.emit_feeback(lfeedback);
         }

         // Mutate weights
         bool overflowed = false;
         //--- mutate gate weight base
         overflowed |= mutate_weight(gate.weight_base, gate.mut_prob_neg, gate.mut_prob_pos);
         //--- mutate links weight
         for (auto& link : links) {
            overflowed |= mutate_weight(link.weight, link.mut_prob_neg, link.mut_prob_pos);
         }
         if (overflowed) {
            downscale_weights();
         }
      }
      __declspec(noinline) void downscale_weights() {
         for (auto& link : links) {
            auto prev_weight = link.weight;
            link.weight = round(double(prev_weight) * 0.5);
         }
         gate.weight_base = round(double(gate.weight_base) * 0.5);
      }
      bool mutate_weight(weight_t& weight, Scalar& mut_prob_neg, Scalar& mut_prob_pos) {
         bool has_overflowed = false;
         bool has_mut = false;
         if (mut_prob_neg >= 0 && random_unsigned() < mut_prob_neg) {
            weight--;
            has_mut = true;
            if (weight < WeightMin) has_overflowed = true;
         }
         if (mut_prob_pos >= 0 && random_unsigned() < mut_prob_pos) {
            weight++;
            has_mut = true;
            if (weight > WeightMax) has_overflowed = true;
         }
         if (has_mut) {
            mut_prob_neg *= 0.5;
            mut_prob_pos *= 0.5;
            mut_prob_pos = clamp<Scalar>(mut_prob_pos, 0, 1);
            mut_prob_neg = clamp<Scalar>(mut_prob_neg, 0, 1);
         }
         return has_overflowed;
      }
      static Scalar random_unsigned() {
         return distribution_unsigned(generator);
      }
      static Scalar random_signed() {
         return distribution_signed(generator);
      }
   };

   struct GateLayer : std::vector<GateObject> {
      int level = 0;
      GateLayer(int count, int level)
         : vector(count), level(level) {
      }
      void emit_feeback(std::vector<Scalar> feedbacks) {
         if (feedbacks.size() != this->size()) throw;

         int index = 0;
         for (auto& gate : (*this)) {
            gate.emit_feeback(feedbacks[index]);
            index++;
         }
      }
      void write_vec8(std::vector<uint8_t> values) {
         if (values.size() * 8 != this->size()) throw;

         int index = 0;
         for (auto& gate : (*this)) {
            uint8_t mask = 1 << (index % 8);
            gate.state = values[index / 8] & mask;
            index++;
         }
      }
      void initialize() {
         for (auto& gate : (*this)) {
            gate.initialize();
         }
      }
      void compute_forward() {
         for (auto& gate : (*this)) {
            gate.compute_forward();
         }
      }
      void compute_backward() {
         for (auto& gate : (*this)) {
            gate.compute_backward();
         }
      }
      GateObject& get(int state_index) {
         return (*this)[state_index];
      }
   };

   struct GateObjectModel {
      std::vector<std::unique_ptr<GateLayer>> layers;
      GateLayer* add_layer(int count, int level) {
         auto layer = new GateLayer(count, level);
         this->layers.push_back(std::unique_ptr<GateLayer>(layer));
         return layer;
      }
      void connect_layer(GateLayer* from_layer, GateLayer* to_layer) {
         if (from_layer->level >= to_layer->level) throw;

         for (auto& from_gate : *from_layer) {
            for (auto& to_gate : *to_layer) {
               to_gate.links.push_back(from_gate);
            }
         }
      }
      void initialize() {
         std::sort(this->layers.begin(), this->layers.end(), [](const std::unique_ptr<GateLayer>& a, const std::unique_ptr<GateLayer>& b) {
            return a->level < b->level;
            });
         for (int i = 0; i < this->layers.size(); i++) {
            auto layer = this->layers[i].get();
            layer->initialize();
         }
      }
      void compute_forward() {
         for (int i = 0; i < this->layers.size(); i++) {
            auto layer = this->layers[i].get();
            layer->compute_forward();
         }
      }
      void compute_backward() {
         for (int i = this->layers.size() - 1; i >= 0; i--) {
            auto layer = this->layers[i].get();
            layer->compute_backward();
         }
      }
   };

   namespace Models {
      struct SingleGateImage2DModel : IImage2DTrainable {
         GateObjectModel model;
         GateLayer& inputs;
         GateLayer& outputs;
         SingleGateImage2DModel() :
            inputs(*model.add_layer(16, 0)),
            outputs(*model.add_layer(1, 1))
         {
            model.connect_layer(&inputs, &outputs);
            model.initialize();
         }
         bool estimate_pixel(uint8_t i, uint8_t j) override {
            inputs.write_vec8({ i, j });
            model.compute_forward();
            return outputs[0].state;
         }
         bool train_pixel(uint8_t i, uint8_t j, bool expected) override {
            inputs.write_vec8({ i, j });
            model.compute_forward();
            auto r = outputs[0].state;

            Scalar feedback = (r == expected) ? 1.0f : -1.0f;
            outputs.emit_feeback({ feedback });
            model.compute_backward();

            return outputs[0].state;
         }
      };
      struct HiddenLayerImage2DModel : IImage2DTrainable {
         GateObjectModel model;
         GateLayer& inputs;
         GateLayer& outputs;
         HiddenLayerImage2DModel() :
            inputs(*model.add_layer(16, 0)),
            outputs(*model.add_layer(1, 2))
         {
            auto hidden_layer = model.add_layer(4, 1);
            model.connect_layer(&inputs, hidden_layer);
            model.connect_layer(hidden_layer, &outputs);
            model.initialize();
         }
         bool estimate_pixel(uint8_t i, uint8_t j) override {
            inputs.write_vec8({ i, j });
            model.compute_forward();
            return outputs[0].state;
         }
         bool train_pixel(uint8_t i, uint8_t j, bool expected) override {
            inputs.write_vec8({ i, j });
            model.compute_forward();
            auto r = outputs[0].state;

            Scalar feedback = (r == expected) ? 1.0f : -1.0f;
            outputs.emit_feeback({ feedback });
            model.compute_backward();

            return outputs[0].state;
         }
      };
   }
}
