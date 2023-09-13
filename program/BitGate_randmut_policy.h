#pragma once

#include "./BitGateMemory.h"
#include "./BitGateModel.h"
#include <algorithm>

extern void print_line(int line, const char* pattern, ...);

namespace ins {

   typedef float Scalar;



   struct RandMutationGatePolicy {

      struct Gate {
         struct Parameter {
            int16_t threshold;
         };
         struct Stats : Probabilistic::GateStats {
            // Accumulated feedback
            Scalar feedback_signal = 0;
            void write_feedback(Scalar signal) {
               this->feedback_signal += signal;
            }
         };
      };

      struct Link {
         struct Parameter {
            int16_t weight_0;
            int16_t weight_1;
         };
         struct Stats {
            Scalar mut_prob_0_neg = 0;
            Scalar mut_prob_0_pos = 0;
            Scalar mut_prob_1_neg = 0;
            Scalar mut_prob_1_pos = 0;
         };
      };


      typedef BitMesh<RandMutationGatePolicy>::GateInstance GateInstance;

      static void initialize(GateInstance& gate, size_t gates_width) {

         // Initiate gate links param
         for (auto link : gate) {
            auto& lparam = link.param();
#if 1
            lparam.weight_1 = rand() - rand();
            lparam.weight_0 = rand() - rand();
#else
            lparam.weight_1 = 0;
            lparam.weight_0 = 0;
#endif
         }

         // Initiate gate core param
         auto& stats = gate.stats();
         auto& param = gate.param();
         param.threshold = 1024;
      }

      static void compute_forward(GateInstance& gate) {

         int32_t acc = 0;
         for (auto link : gate) {
            auto& lparam = link.param();
            if (link.get()) acc += lparam.weight_1;
            else acc += lparam.weight_0;
         }

         auto& param = gate.param();
         gate.set(acc > param.threshold);
      }

      static void compute_backward(GateInstance& gate) {

         // Flush integrated feedback signal
         auto& stats = gate.stats();
         auto feedback = stats.feedback_signal;
         stats.feedback_signal = 0;

         // Integrate feedback to gate stats
         auto gate_state = gate.get();
         stats.add(gate_state, feedback);

         // Compute links weights sum
         int32_t links_weights_sum = 0;
         for (auto link : gate) {
            auto link_state = link.get();
            auto& lparam = link.param();
            if (link_state) links_weights_sum += abs(lparam.weight_1);
            else links_weights_sum += abs(lparam.weight_0);
         }

         Scalar feedback_prob = 0.001 * std::abs(feedback * feedback);

         // Integrate feedback to links stats
         Scalar weight_factor = links_weights_sum > 0 ? 1.0 / Scalar(links_weights_sum) : 0;
         int32_t links_feedback = 0;
         for (auto link : gate) {
            auto link_state = link.get();
            auto& lstats = link.stats();
            auto& lparam = link.param();

            // Compute link feedback
            Scalar lfeedback = 0;
            if (link_state == 1) {
               lfeedback = feedback * lparam.weight_1 * weight_factor;
               if (feedback > 0) {
                  lstats.mut_prob_1_neg -= feedback_prob;
                  lstats.mut_prob_1_pos -= feedback_prob;
               }
               else {
                  if (gate_state == 1) {
                     lstats.mut_prob_1_neg += feedback_prob;
                  }
                  else {
                     lstats.mut_prob_1_pos += feedback_prob;
                  }
               }
               lstats.mut_prob_1_pos = clamp<Scalar>(lstats.mut_prob_1_pos, 0, 1);
               lstats.mut_prob_1_neg = clamp<Scalar>(lstats.mut_prob_1_neg, 0, 1);
            }
            else {
               lfeedback = feedback * lparam.weight_0 * weight_factor;
               if (feedback > 0) {
                  lstats.mut_prob_0_neg -= feedback_prob;
                  lstats.mut_prob_0_pos -= feedback_prob;
               }
               else {
                  if (gate_state == 0) {
                     lstats.mut_prob_0_neg += feedback_prob;
                  }
                  else {
                     lstats.mut_prob_0_pos += feedback_prob;
                  }
               }
               lstats.mut_prob_0_pos = clamp<Scalar>(lstats.mut_prob_0_pos, 0, 1);
               lstats.mut_prob_0_neg = clamp<Scalar>(lstats.mut_prob_0_neg, 0, 1);
            }

            // Dispatch feedback to link input stats
            link.emit_feeback(lfeedback);
         }

         // Mutate
         for (auto link : gate) {
            auto& lstats = link.stats();
            auto& lparam = link.param();
            bool has_mut = false;
            if (random() < lstats.mut_prob_0_neg) {
               lparam.weight_0--;
               has_mut = true;
            }
            if (random() < lstats.mut_prob_0_pos) {
               lparam.weight_0++;
               has_mut = true;
            }
            if (random() < lstats.mut_prob_1_neg) {
               lparam.weight_1--;
               has_mut = true;
            }
            if (random() < lstats.mut_prob_0_pos) {
               lparam.weight_1++;
               has_mut = true;
            }
            if (has_mut) {
               Scalar factor = 0.0;
               lstats.mut_prob_0_neg *= factor;
               lstats.mut_prob_0_pos *= factor;
               lstats.mut_prob_1_neg *= factor;
               lstats.mut_prob_1_pos *= factor;
            }
         }
      }

      static Scalar random() {
         return Scalar(rand()) / Scalar(RAND_MAX);
      }

      static void mutate_forward(GateInstance& gate) {

      }

      static void mutate_backward(GateInstance& gate) {

      }
   };
}
