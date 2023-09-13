#pragma once

#include "./BitGateMemory.h"
#include "./BitGateModel.h"

extern void print_line(int line, const char* pattern, ...);

namespace ins {

   struct BasicGatePolicy {
      // Method: Integrate progressivle correction on weights, and apply it on weights when correction became great

      struct Gate {
         struct Parameter {
            int16_t scaleL2;
         };
         struct Stats {
            int32_t feedback_signal = 0;
            void write_feedback(int32_t signal) {
               this->feedback_signal += signal;
            }
         };
      };

      struct Link {
         struct Parameter {
            Scalar weight_0;
            Scalar weight_1;
         };
         struct Stats {
         };
      };

      typedef BitMesh<BasicGatePolicy>::GateInstance GateInstance;

      static constexpr Scalar threshold = 10000;
      static constexpr bool use_implicit_threshold_regulation = true;

      static Scalar get_links_magnitude(GateInstance& gate) {
         Scalar sum = 0;
         for (auto link : gate) {
            auto& lparam = link.param();
            sum += (lparam.weight_1 * lparam.weight_1);
            sum += (lparam.weight_0 * lparam.weight_0);
         }
         return sum;
      }

      static void initialize(GateInstance& gate, size_t gates_width) {
         for (auto link : gate) {
            auto& lparam = link.param();
            lparam.weight_1 = rand()- rand();
            lparam.weight_0 = rand()-rand();
         }
      }
      static void compute_forward(GateInstance& gate) {
         Scalar acc = 0;
         for (auto link : gate) {
            auto& lparam = link.param();
            if (link.get()) acc += lparam.weight_1;
            else acc += lparam.weight_0;
         }
         gate.set(acc > threshold);
      }

      static void compute_backward(GateInstance& gate) {

         // Flush integrated feedback signal
         auto& stats = gate.stats();
         auto& param = gate.param();
         auto feedback = stats.feedback_signal;
         stats.feedback_signal = 0;

         if (feedback >= 0) {
            return; // Nothing to correct
         }

         // Integrate gates values
         Scalar delta_acc = 0;
         for (auto link : gate) {
            auto& lstats = link.stats();
            auto& lparam = link.param();
            Scalar lvalue;
            if (link.get()) {
               lvalue = lparam.weight_1;
            }
            else {
               lvalue = lparam.weight_0;
            }
            delta_acc += lvalue;
         }

         if (gate.get()) {
            if (delta_acc < threshold) return; // Nothing to correct
         }
         else {
            if (delta_acc > threshold) return; // Nothing to correct
         }

         auto delta = (Scalar(threshold) - delta_acc) / Scalar(gate.page->gates_width);

         static int c = 0;
         c++;
         if (c <400) {
            delta *= 0.1;
         }
         else if (c < 800) {
            delta *= 0.01;
         }
         else {
            delta *= 0.001;
         }

         // Integrate gates values
         for (auto link : gate) {
            auto& lstats = link.stats();
            auto& lparam = link.param();
            if (link.get()) {
               lparam.weight_1 += delta;
            }
            else {
               lparam.weight_0 += delta;
            }
         }

         /*Scalar weight_threshold = 0;
         Scalar weight_1_sum = 0;
         Scalar weight_0_sum = 0;
         for (auto link : gate) {
            auto& lparam = link.param();
            if (use_implicit_threshold_regulation) {
               if (lparam.weight_1 > 0 && lparam.weight_0 > 0) {
                  if (lparam.weight_1 < lparam.weight_0) {
                     weight_threshold += lparam.weight_1;
                     lparam.weight_0 -= lparam.weight_1;
                     lparam.weight_1 = 0;
                  }
                  else {
                     weight_threshold += lparam.weight_0;
                     lparam.weight_1 -= lparam.weight_0;
                     lparam.weight_0 = 0;
                  }
               }
               else if (lparam.weight_1 < 0 && lparam.weight_0 < 0) {
                  if (lparam.weight_1 > lparam.weight_0) {
                     weight_threshold += lparam.weight_1;
                     lparam.weight_0 -= lparam.weight_1;
                     lparam.weight_1 = 0;
                  }
                  else {
                     weight_threshold += lparam.weight_0;
                     lparam.weight_1 -= lparam.weight_0;
                     lparam.weight_0 = 0;
                  }
               }
            }
            weight_1_sum = lparam.weight_1;
            weight_0_sum = lparam.weight_0;
         }
         print_line(0, "magnitude: %lg", get_links_magnitude(gate));

         // Integrate feedback to gates_links stats
         for (auto link : gate) {
            auto& lparam = link.param();

            // Compute link feedback
            auto lfeedback = weight_1_sum ? (int64_t(feedback) * lparam.weight_1) / weight_1_sum : 0;
            if (lparam.weight_1 < 0) lfeedback = -lfeedback;

            // Dispatch feedback to link input stats
            link.emit_feeback(lfeedback);
         }*/

      }

      static void mutate_forward(GateInstance& gate) {

      }

      static void mutate_backward(GateInstance& gate) {

      }
   };
}
