#pragma once

#include "./BitGateMemory.h"
#include "./BitGateModel.h"

namespace ins {

   struct BitGatePolicy {

      static void compute_forward(BitGateMemory* mem, BitGatePage* page, int gate_index, int param_index) {
         auto param_end = param_index + page->gates_width;
         auto scaleL2 = page->units_param[param_index].gate.scaleL2;

         // Accumulate links value
         uint64_t acc = 0;
         for (auto link_index = param_index + 1; link_index < param_end; link_index++) {
            auto& lparam = page->units_param[link_index].link;

            // Integrate bit to gate value
            auto bit_state = mem->get_gate_state(page->gates_links[link_index]);
            if (bit_state) {
               acc += lparam.weight_1;
            }
            else {
               acc += lparam.weight_0;
            }
         }

         // Eval gate activation based on accumulate value
         auto active = acc >> scaleL2;
         page->set_gate_state(gate_index, active);
      }

      static void compute_backward(BitGateMemory* mem, BitGatePage* page, int gate_index, int param_index) {
         auto param_end = param_index + page->gates_width;

         // Flush integrated feedback signal
         auto& stats = page->units_stats[param_index].gate;
         auto feedback = stats.feedback_signal;
         stats.feedback_signal = 0;

         // Integrate feedback to gate stats
         auto bit_state = page->get_gate_state(gate_index);
         if (bit_state) {
            if (feedback > 0) {
               stats.R_C1 += feedback;
            }
            else {
               stats.P_C1 += -feedback;
            }
         }
         else {
            if (feedback > 0) {
               stats.R_C0 += feedback;
            }
            else {
               stats.P_C0 += -feedback;
            }
         }

         int32_t heaviest_index = 0;
         int32_t heaviest_weight = 0;

         // Integrate feedback to gates_links stats
         int32_t links_weights_sum = stats.weights_sum;
         int32_t links_feedback = 0;
         for (auto link_index = param_index + 1; link_index < param_end; link_index++) {
            auto link_state = mem->get_gate_state(page->gates_links[link_index]);
            auto& lstats = page->units_stats[link_index].link;
            auto& lparam = page->units_param[link_index].link;

            // Update heaviest link search
            if (heaviest_weight < lparam.weight_1) {
               heaviest_index = link_index;
               heaviest_weight = lparam.weight_1;
            }

            // Compute link feedback
            auto lfeedback = lparam.weight_1 ? (int64_t(feedback) * lparam.weight_1) / links_weights_sum : 0;
            if (lparam.weight_1 < 0) lfeedback = -lfeedback;
            links_feedback += lfeedback;

            // Integrate feedback to link stats
            if (bit_state) {
               if (lfeedback > 0) {
                  if (link_state) lstats.R_C1_I1 += lfeedback;
                  else lstats.R_C1_I0 += lfeedback;
               }
               else {
                  if (link_state) lstats.P_C1_I1 += -lfeedback;
                  else lstats.P_C1_I0 += -lfeedback;
               }
            }
            else {
               if (lfeedback > 0) {
                  if (link_state) lstats.R_C0_I1 += lfeedback;
                  else lstats.R_C0_I0 += lfeedback;
               }
               else {
                  if (link_state) lstats.P_C0_I1 += -lfeedback;
                  else lstats.P_C0_I0 += -lfeedback;
               }
            }

            // Dispatch feedback to link input stats
            mem->emit_gate_feeback(page->gates_links[link_index], lfeedback);
         }

         if (links_feedback != feedback) {
            _ASSERT(std::abs(links_feedback) < std::abs(feedback));
            auto link_index = heaviest_index;
            auto link_state = mem->get_gate_state(page->gates_links[link_index]);
            auto& lstats = page->units_stats[link_index].link;

            // Compute link feedback
            auto lfeedback = feedback - links_feedback;
            links_feedback += lfeedback;

            // Integrate feedback to link stats
            if (bit_state) {
               if (lfeedback > 0) {
                  if (link_state) lstats.R_C1_I1 += lfeedback;
                  else lstats.R_C1_I0 += lfeedback;
               }
               else {
                  if (link_state) lstats.P_C1_I1 += -lfeedback;
                  else lstats.P_C1_I0 += -lfeedback;
               }
            }
            else {
               if (lfeedback > 0) {
                  if (link_state) lstats.R_C0_I1 += lfeedback;
                  else lstats.R_C0_I0 += lfeedback;
               }
               else {
                  if (link_state) lstats.P_C0_I1 += -lfeedback;
                  else lstats.P_C0_I0 += -lfeedback;
               }
            }

            // Dispatch feedback to link input stats
            mem->emit_gate_feeback(page->gates_links[link_index], lfeedback);
         }
         _ASSERT(links_feedback == feedback);

         // Mutate (consume a part of penalty bugdet to mutate weight)
         auto P_Cx = stats.P_C0 + stats.P_C1;
         auto R_Cx = stats.R_C0 + stats.R_C1;
         if (P_Cx > 100000) {

            // Consume mutation bugdet
            for (auto link_index = param_index + 1; link_index < param_end; link_index++) {
               auto& lstats = page->units_stats[link_index].link;
               //-- R_Cx_Iy : reward when consencus at x and input at y
               lstats.R_C0_I0 >>= 1;
               lstats.R_C0_I1 >>= 1;
               lstats.R_C1_I0 >>= 1;
               lstats.R_C1_I1 >>= 1;
               //-- P_Cx_Iy : penalty when consencus at x and input at y
               lstats.P_C0_I0 >>= 1;
               lstats.P_C0_I1 >>= 1;
               lstats.P_C1_I0 >>= 1;
               lstats.P_C1_I1 >>= 1;
            }
            //-- R_Cx : reward when consencus at x
            stats.R_C0 >>= 1;
            stats.R_C1 >>= 1;
            //-- P_Cx : penalty when consencus at x
            stats.P_C0 >>= 1;
            stats.P_C1 >>= 1;

            // Mutation bugdet
            //printf("mutate: {\n");
            for (auto link_index = param_index + 1; link_index < param_end; link_index++) {
               auto link_state = mem->get_gate_state(page->gates_links[link_index]);
               auto& lstats = page->units_stats[link_index].link;

               auto lP_Cx = lstats.P_C0_I0 + lstats.P_C0_I1 + lstats.P_C1_I0 + lstats.P_C1_I1;
               auto lR_Cx = lstats.R_C0_I0 + lstats.R_C0_I1 + lstats.R_C1_I0 + lstats.R_C1_I1;

               //printf("link: r=%g P=%g R=%g\n", float(lP_Cx) / float(lR_Cx), float(lP_Cx) / float(P_Cx), float(lR_Cx) / float(R_Cx));
            }
            //printf("}\n");
         }
      }

      static void mutate_forward(BitGateMemory* mem, BitGatePage* page, int gate_index, int param_index) {

      }

      static void mutate_backward(BitGateMemory* mem, BitGatePage* page, int gate_index, int param_index) {

      }
   };
}
