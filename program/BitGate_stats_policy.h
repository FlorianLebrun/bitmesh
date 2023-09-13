#pragma once

#include "./BitGateMemory.h"
#include "./BitGateModel.h"

extern void print_line(int line, const char* pattern, ...);

namespace ins {

   typedef float Scalar;

   struct StatsGatePolicy {
      // Method: 
      //   P1_acc: Mean accumulate value when penalized with output on 1
      //   P0_acc: Mean accumulate value when penalized with output on 0
      //   R1_acc: Mean accumulate value when rewarded with output on 1
      //   R0_acc: Mean accumulate value when rewarded with output on 0
      // Optimize requirement:
      //   P1_acc > Threshold, shall became <= Threshold
      //   R1_acc > Threshold, shall minimize it move
      //   P0_acc < Threshold, shall became >= Threshold
      //   R0_acc < Threshold, shall minimize it move
      // Optimized lost function:
      //   L(W') = (P1_acc(W')-Threshold)² + (P0_acc(W')-Threshold)² + (R1_acc(W')-R1_acc(W))² + (R0_acc(W')-R0_acc(W))²

      struct Gate {
         struct Parameter {
            int16_t threshold;
         };
         struct Stats : Probabilistic::GateStats {
            // Accumulated feedback
            int32_t feedback_signal = 0;
            void write_feedback(int32_t signal) {
               this->feedback_signal += signal;
            }
         };
      };

      struct Link {
         struct Parameter {
            int16_t weight_0;
            int16_t weight_1;
         };
         struct Stats : Probabilistic::LinkStats {
            // Paremeters
            uint32_t drift = 0;          // mutation asymetry factor (~ around 1.0)
         };
      };


      typedef BitMesh<StatsGatePolicy>::GateInstance GateInstance;

      static void initialize(GateInstance& gate, size_t gates_width) {

         // Initiate gate links param
         for (auto link : gate) {
            auto& lparam = link.param();
            lparam.weight_1 = 0;
            lparam.weight_0 = 0;
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
         int32_t heaviest_index = 0;
         int32_t heaviest_weight = 0;

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
            if (link_state) links_weights_sum += lparam.weight_1;
            else links_weights_sum += lparam.weight_0;
         }

         // Integrate feedback to links stats
         int32_t links_feedback = 0;
         for (auto link : gate) {
            auto link_state = link.get();
            auto& lstats = link.stats();
            auto& lparam = link.param();

            // Update heaviest link search
            if (heaviest_weight < lparam.weight_1) {
               heaviest_index = link.link_index;
               heaviest_weight = lparam.weight_1;
            }

            // Compute link feedback
            auto lfeedback = lparam.weight_1 ? (int64_t(feedback) * lparam.weight_1) / links_weights_sum : 0;
            if (lparam.weight_1 < 0) lfeedback = -lfeedback;
            links_feedback += lfeedback;

            // Integrate feedback to link stats
            lstats.add(gate_state, link_state, lfeedback);

            // Dispatch feedback to link input stats
            link.emit_feeback(lfeedback);
         }

         if (links_feedback != feedback) {
            // _ASSERT(std::abs(links_feedback) < std::abs(feedback));
            auto link = gate.at(heaviest_index);
            auto link_state = link.get();
            auto& lstats = link.stats();

            // Compute link feedback
            auto lfeedback = feedback - links_feedback;
            links_feedback += lfeedback;

            // Integrate feedback to link stats
            lstats.add(gate_state, link_state, lfeedback);

            // Dispatch feedback to link input stats
            link.emit_feeback(lfeedback);
         }
         _ASSERT(links_feedback == feedback);

         // Mutate (consume a part of penalty bugdet to mutate weight)
         auto Px = stats.P0 + stats.P1;
         auto Rx = stats.R0 + stats.R1;
         if (Px > 100000) {
            int line = 2;

            // Consume mutation bugdet
            for (auto link : gate) {
               auto& lstats = link.stats();
               //-- Rx_Iy : reward when consencus at x and input at y
               lstats.R0_I0 >>= 1;
               lstats.R0_I1 >>= 1;
               lstats.R1_I0 >>= 1;
               lstats.R1_I1 >>= 1;
               //-- Px_Iy : penalty when consencus at x and input at y
               lstats.P0_I0 >>= 1;
               lstats.P0_I1 >>= 1;
               lstats.P1_I0 >>= 1;
               lstats.P1_I1 >>= 1;
            }
            //-- Rx : reward when consencus at x
            stats.R0 >>= 1;
            stats.R1 >>= 1;
            //-- Px : penalty when consencus at x
            stats.P0 >>= 1;
            stats.P1 >>= 1;

            // Mutation bugdet
            //printf("mutate: {\n");
            int32_t bias = 0;
            for (auto link : gate) {
               auto link_state = link.get();
               auto& lparam = link.param();
               auto& lstats = link.stats();

               auto lPx = lstats.Px();
               auto lRx = lstats.Rx();

               print_line(line, "link: Px=%d Rx=%d", lPx, lRx);
               line++;
            }
            print_line(line, "> mutate: Px=%d Rx=%d", Px, Rx);
         }
      }

      static void mutate_forward(GateInstance& gate) {

      }

      static void mutate_backward(GateInstance& gate) {

      }
   };
}
