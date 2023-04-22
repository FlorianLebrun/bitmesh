#pragma once

#include "./BitGateMemory.h"
#include "./BitGateModel.h"

namespace ins {

   struct BitGatePolicy {

      static void compute_forward(BitGateMemory* mem, BitGatePage* page, int gate_index, int param_index) {
         auto param_end = param_index + page->gates_width;
         auto scaleL2 = page->params_values[param_index];
         param_index++;

         // Accumulate links value
         uint64_t acc = 0;
         while (param_index < param_end) {
            auto link_value = mem->get_gate_state(page->gates_links[param_index]);
            if (link_value) {
               auto weigth = page->params_values[param_index];
               acc += weigth;
            }
            param_index++;
         }

         // Eval gate activation based on accumulate value
         auto active = acc >> scaleL2;
         page->set_gate_state(gate_index, active);
      }

      static void compute_backward(BitGateMemory* mem, BitGatePage* page, int gate_index, int param_index) {
         auto param_end = param_index + page->gates_width;

         // Flush integrated feedback signal
         auto& stats = page->params_stats[param_index].gate_stats;
         auto feedback = stats.feedback_signal;
         stats.feedback_signal = 0;
         param_index++;

         // Integrate feedback to gate stats
         auto bit_state = page->get_gate_state(gate_index);
         if (feedback > 0) {
            if (bit_state) stats.R_C1 += feedback;
            else stats.R_C0 += feedback;
         }
         else {
            if (bit_state) stats.P_C1 += -feedback;
            else stats.P_C0 += -feedback;
         }


         int32_t heaviest_index = 0;
         int32_t heaviest_weight = 0;

         // Integrate feedback to gates_links stats
         int32_t links_weights_sum = stats.weights_sum;
         int32_t links_feedback = 0;
         while (param_index < param_end) {
            auto link_state = mem->get_gate_state(page->gates_links[param_index]);
            auto& lstats = page->params_stats[param_index].link_stats;
            auto lweight = page->params_values[param_index];

            // Update heaviest link search
            if (heaviest_weight < lweight) {
               heaviest_index = param_index;
               heaviest_weight = lweight;
            }

            // Compute link feedback
            auto lfeedback = lweight ? (int64_t(feedback) * lweight) / links_weights_sum : 0;
            if (lweight < 0) lfeedback = -lfeedback;
            links_feedback += lfeedback;

            // Integrate feedback to link stats
            if (lfeedback > 0) {
               if (bit_state) {
                  if (link_state) lstats.R_C1_I1 += lfeedback;
                  else lstats.R_C1_I0 += lfeedback;
               }
               else {
                  if (link_state) lstats.R_C0_I1 += lfeedback;
                  else lstats.R_C0_I0 += lfeedback;
               }
            }
            else {
               if (bit_state) {
                  if (link_state) lstats.P_C1_I1 += -lfeedback;
                  else lstats.P_C1_I0 += -lfeedback;
               }
               else {
                  if (link_state) lstats.P_C0_I1 += -lfeedback;
                  else lstats.P_C0_I0 += -lfeedback;
               }
            }

            // Dispatch feedback to link input stats
            mem->emit_gate_feeback(page->gates_links[param_index], lfeedback);

            param_index++;
         }

         if (links_feedback != feedback) {
            _ASSERT(std::abs(links_feedback) < std::abs(feedback));
            auto param_index = heaviest_index;
            auto link_state = mem->get_gate_state(page->gates_links[param_index]);
            auto& lstats = page->params_stats[param_index].link_stats;

            // Compute link feedback
            auto lfeedback = feedback - links_feedback;
            links_feedback += lfeedback;

            // Integrate feedback to link stats
            if (lfeedback > 0) {
               if (bit_state) {
                  if (link_state) lstats.R_C1_I1 += lfeedback;
                  else lstats.R_C1_I0 += lfeedback;
               }
               else {
                  if (link_state) lstats.R_C0_I1 += lfeedback;
                  else lstats.R_C0_I0 += lfeedback;
               }
            }
            else {
               if (bit_state) {
                  if (link_state) lstats.P_C1_I1 += -lfeedback;
                  else lstats.P_C1_I0 += -lfeedback;
               }
               else {
                  if (link_state) lstats.P_C0_I1 += -lfeedback;
                  else lstats.P_C0_I0 += -lfeedback;
               }
            }

            // Dispatch feedback to link input stats
            mem->emit_gate_feeback(page->gates_links[param_index], lfeedback);
         }

         // Mutate

         _ASSERT(links_feedback == feedback);
      }

      static void mutate_forward(BitGateMemory* mem, BitGatePage* page, int gate_index, int param_index) {
         auto param_end = param_index + page->gates_width;
         auto scaleL2 = page->params_values[param_index];
         auto& stats = page->params_stats[param_index].gate_stats;
         param_index++;

         // TODO: mutate based on stats

         // Mutate gates links
         while (param_index < param_end) {
            auto weight = page->params_values[param_index];
            auto& lstats = page->params_stats[param_index].link_stats;

            // TODO: mutate based on stats & lstats

            param_index++;
         }
      }

      static void mutate_backward(BitGateMemory* mem, BitGatePage* page, int gate_index, int param_index) {
         auto param_end = param_index + page->gates_width;
         auto scaleL2 = page->params_values[param_index];
         auto& stats = page->params_stats[param_index].gate_stats;
         param_index++;

         // TODO: mutate based on stats

         // Mutate gates links
         while (param_index < param_end) {
            auto weight = page->params_values[param_index];
            auto& lstats = page->params_stats[param_index].link_stats;

            // TODO: mutate based on stats & lstats

            param_index++;
         }
      }
   };
}
