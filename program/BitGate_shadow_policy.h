#pragma once

#include "./BitGateMemory.h"
#include "./BitGateModel.h"

extern void print_line(int line, const char* pattern, ...);

namespace ins {
   typedef float Scalar;

   struct ShadowGatePolicy {
      // Method: Optimize shadow gate and flush it into real gate, when better than real gate

      struct Gate {
         struct Parameter {
            Scalar scaleL2 = 0;
         };
         struct Stats {
            Parameter shadow;
            int32_t feedback_signal = 0;
            void write_feedback(int32_t signal) {
               this->feedback_signal += signal;
            }
         };
      };

      struct Link {
         struct Parameter {
            Scalar weight_0 = 0;
            Scalar weight_1 = 0;
         };
         struct Stats {
            Parameter shadow;
         };
      };

      typedef BitMesh<ShadowGatePolicy>::GateInstance GateInstance;

      static void initialize(GateInstance& gate, size_t gates_width) {

      }

      static void compute_forward(GateInstance& gate) {

      }

      static void compute_backward(GateInstance& gate) {
      }

      static void mutate_forward(GateInstance& gate) {

      }

      static void mutate_backward(GateInstance& gate) {

      }
   };
}
