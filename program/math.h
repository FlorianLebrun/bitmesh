#pragma once

#include <vector>
#include <algorithm>
#include <random>

namespace ins {
   
   typedef float Scalar;

   template <class _Ty>
   constexpr const _Ty& clamp(const _Ty& x, const _Ty& _min, const _Ty& _max) {
      return x < _min ? _min : (x > _max ? _max : x);
   }

   struct IImage2DModel {
      virtual bool estimate_pixel(uint8_t i, uint8_t j) = 0;
      void print_image(int at_line = 4);

   };

   struct IImage2DTrainable : IImage2DModel {
      virtual bool train_pixel(uint8_t i, uint8_t j, bool expected) = 0;
   };

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

}
