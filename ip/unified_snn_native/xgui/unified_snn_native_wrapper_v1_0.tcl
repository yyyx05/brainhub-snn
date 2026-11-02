# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "ACCUMULATE_SCNN_SAMPLE_LOGITS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "BATCH_COUNT_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CHANNEL_COUNT_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CHANNEL_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CLASSIFIER_CAPTURE_MAX_BATCH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CLASSIFIER_CAPTURE_MAX_TIMESTEPS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CLASS_COUNT_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CLASS_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CODE_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "ENABLE_ONLINE_LEARNING" -parent ${Page_0}
  ipgui::add_param $IPINST -name "ENABLE_SCNN_CONV8_LEARNING" -parent ${Page_0}
  ipgui::add_param $IPINST -name "ENABLE_SCNN_FRAME_SERVICE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "ENABLE_SRNN_WH1_LEARNING" -parent ${Page_0}
  ipgui::add_param $IPINST -name "ENGINES" -parent ${Page_0}
  ipgui::add_param $IPINST -name "FC_MAX_HIDDEN" -parent ${Page_0}
  ipgui::add_param $IPINST -name "FC_MAX_OUTPUTS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "FEATURE_ADDR_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "FEATURE_COUNT_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "FEATURE_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "FEATURE_WORD_DEPTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "FRAME_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "INTERNAL_MODULATION_BATCH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "KERNEL_SIZE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "LANES" -parent ${Page_0}
  ipgui::add_param $IPINST -name "MAX_BATCH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "MAX_CLASSES" -parent ${Page_0}
  ipgui::add_param $IPINST -name "MAX_FEATURES" -parent ${Page_0}
  ipgui::add_param $IPINST -name "MAX_INPUT_CHANNELS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "MAX_OUTPUT_CHANNELS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "MAX_TIMESTEPS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "MEMORY_ADDR_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "MEMORY_DEPTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "OBS_INPUT_CHANNELS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "OBS_OUTPUT_CHANNELS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "PARAM_ADDR_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "PARAM_DEPTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "POOLED_SPATIAL" -parent ${Page_0}
  ipgui::add_param $IPINST -name "PROFILE_ID_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "Q16_ADDR_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "Q64_ADDR_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "RESIDUAL_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SAMPLE_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SCNN_WEIGHT_FRACTIONAL_BITS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SELECTED_LAYER" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SPATIAL" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SRNN_BACKBONE_MAX" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SRNN_BACKBONE_PARAM_ADDR_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SRNN_BACKBONE_PARAM_DEPTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SRNN_BATCH_MAX" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SRNN_CLASS_MAX" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SRNN_FEEDFORWARD_SOURCE_MAX" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SRNN_HIDDEN_MAX" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SRNN_M2_BACKBONE_MAX" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SRNN_M2_BATCH_MAX" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SRNN_M2_CLASS_MAX" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SRNN_M2_HIDDEN_MAX" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SRNN_M2_PARAM_DEPTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SRNN_M2_TIMESTEP_MAX" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SRNN_RECURRENT_SOURCE_MAX" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SRNN_TIMESTEP_MAX" -parent ${Page_0}
  ipgui::add_param $IPINST -name "TAG_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "TENSOR_ID_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "TIMESTEP_COUNT_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "TIMESTEP_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "TOPOLOGY_MASK" -parent ${Page_0}
  ipgui::add_param $IPINST -name "VERSION_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "WEIGHT_ADDR_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "WEIGHT_DEPTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "WORK_ADDR_WIDTH" -parent ${Page_0}


}

proc update_PARAM_VALUE.ACCUMULATE_SCNN_SAMPLE_LOGITS { PARAM_VALUE.ACCUMULATE_SCNN_SAMPLE_LOGITS } {
	# Procedure called to update ACCUMULATE_SCNN_SAMPLE_LOGITS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ACCUMULATE_SCNN_SAMPLE_LOGITS { PARAM_VALUE.ACCUMULATE_SCNN_SAMPLE_LOGITS } {
	# Procedure called to validate ACCUMULATE_SCNN_SAMPLE_LOGITS
	return true
}

proc update_PARAM_VALUE.BATCH_COUNT_WIDTH { PARAM_VALUE.BATCH_COUNT_WIDTH } {
	# Procedure called to update BATCH_COUNT_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BATCH_COUNT_WIDTH { PARAM_VALUE.BATCH_COUNT_WIDTH } {
	# Procedure called to validate BATCH_COUNT_WIDTH
	return true
}

proc update_PARAM_VALUE.CHANNEL_COUNT_WIDTH { PARAM_VALUE.CHANNEL_COUNT_WIDTH } {
	# Procedure called to update CHANNEL_COUNT_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CHANNEL_COUNT_WIDTH { PARAM_VALUE.CHANNEL_COUNT_WIDTH } {
	# Procedure called to validate CHANNEL_COUNT_WIDTH
	return true
}

proc update_PARAM_VALUE.CHANNEL_WIDTH { PARAM_VALUE.CHANNEL_WIDTH } {
	# Procedure called to update CHANNEL_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CHANNEL_WIDTH { PARAM_VALUE.CHANNEL_WIDTH } {
	# Procedure called to validate CHANNEL_WIDTH
	return true
}

proc update_PARAM_VALUE.CLASSIFIER_CAPTURE_MAX_BATCH { PARAM_VALUE.CLASSIFIER_CAPTURE_MAX_BATCH } {
	# Procedure called to update CLASSIFIER_CAPTURE_MAX_BATCH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CLASSIFIER_CAPTURE_MAX_BATCH { PARAM_VALUE.CLASSIFIER_CAPTURE_MAX_BATCH } {
	# Procedure called to validate CLASSIFIER_CAPTURE_MAX_BATCH
	return true
}

proc update_PARAM_VALUE.CLASSIFIER_CAPTURE_MAX_TIMESTEPS { PARAM_VALUE.CLASSIFIER_CAPTURE_MAX_TIMESTEPS } {
	# Procedure called to update CLASSIFIER_CAPTURE_MAX_TIMESTEPS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CLASSIFIER_CAPTURE_MAX_TIMESTEPS { PARAM_VALUE.CLASSIFIER_CAPTURE_MAX_TIMESTEPS } {
	# Procedure called to validate CLASSIFIER_CAPTURE_MAX_TIMESTEPS
	return true
}

proc update_PARAM_VALUE.CLASS_COUNT_WIDTH { PARAM_VALUE.CLASS_COUNT_WIDTH } {
	# Procedure called to update CLASS_COUNT_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CLASS_COUNT_WIDTH { PARAM_VALUE.CLASS_COUNT_WIDTH } {
	# Procedure called to validate CLASS_COUNT_WIDTH
	return true
}

proc update_PARAM_VALUE.CLASS_WIDTH { PARAM_VALUE.CLASS_WIDTH } {
	# Procedure called to update CLASS_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CLASS_WIDTH { PARAM_VALUE.CLASS_WIDTH } {
	# Procedure called to validate CLASS_WIDTH
	return true
}

proc update_PARAM_VALUE.CODE_WIDTH { PARAM_VALUE.CODE_WIDTH } {
	# Procedure called to update CODE_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CODE_WIDTH { PARAM_VALUE.CODE_WIDTH } {
	# Procedure called to validate CODE_WIDTH
	return true
}

proc update_PARAM_VALUE.ENABLE_ONLINE_LEARNING { PARAM_VALUE.ENABLE_ONLINE_LEARNING } {
	# Procedure called to update ENABLE_ONLINE_LEARNING when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ENABLE_ONLINE_LEARNING { PARAM_VALUE.ENABLE_ONLINE_LEARNING } {
	# Procedure called to validate ENABLE_ONLINE_LEARNING
	return true
}

proc update_PARAM_VALUE.ENABLE_SCNN_CONV8_LEARNING { PARAM_VALUE.ENABLE_SCNN_CONV8_LEARNING } {
	# Procedure called to update ENABLE_SCNN_CONV8_LEARNING when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ENABLE_SCNN_CONV8_LEARNING { PARAM_VALUE.ENABLE_SCNN_CONV8_LEARNING } {
	# Procedure called to validate ENABLE_SCNN_CONV8_LEARNING
	return true
}

proc update_PARAM_VALUE.ENABLE_SCNN_FRAME_SERVICE { PARAM_VALUE.ENABLE_SCNN_FRAME_SERVICE } {
	# Procedure called to update ENABLE_SCNN_FRAME_SERVICE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ENABLE_SCNN_FRAME_SERVICE { PARAM_VALUE.ENABLE_SCNN_FRAME_SERVICE } {
	# Procedure called to validate ENABLE_SCNN_FRAME_SERVICE
	return true
}

proc update_PARAM_VALUE.ENABLE_SRNN_WH1_LEARNING { PARAM_VALUE.ENABLE_SRNN_WH1_LEARNING } {
	# Procedure called to update ENABLE_SRNN_WH1_LEARNING when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ENABLE_SRNN_WH1_LEARNING { PARAM_VALUE.ENABLE_SRNN_WH1_LEARNING } {
	# Procedure called to validate ENABLE_SRNN_WH1_LEARNING
	return true
}

proc update_PARAM_VALUE.ENGINES { PARAM_VALUE.ENGINES } {
	# Procedure called to update ENGINES when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ENGINES { PARAM_VALUE.ENGINES } {
	# Procedure called to validate ENGINES
	return true
}

proc update_PARAM_VALUE.FC_MAX_HIDDEN { PARAM_VALUE.FC_MAX_HIDDEN } {
	# Procedure called to update FC_MAX_HIDDEN when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.FC_MAX_HIDDEN { PARAM_VALUE.FC_MAX_HIDDEN } {
	# Procedure called to validate FC_MAX_HIDDEN
	return true
}

proc update_PARAM_VALUE.FC_MAX_OUTPUTS { PARAM_VALUE.FC_MAX_OUTPUTS } {
	# Procedure called to update FC_MAX_OUTPUTS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.FC_MAX_OUTPUTS { PARAM_VALUE.FC_MAX_OUTPUTS } {
	# Procedure called to validate FC_MAX_OUTPUTS
	return true
}

proc update_PARAM_VALUE.FEATURE_ADDR_WIDTH { PARAM_VALUE.FEATURE_ADDR_WIDTH } {
	# Procedure called to update FEATURE_ADDR_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.FEATURE_ADDR_WIDTH { PARAM_VALUE.FEATURE_ADDR_WIDTH } {
	# Procedure called to validate FEATURE_ADDR_WIDTH
	return true
}

proc update_PARAM_VALUE.FEATURE_COUNT_WIDTH { PARAM_VALUE.FEATURE_COUNT_WIDTH } {
	# Procedure called to update FEATURE_COUNT_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.FEATURE_COUNT_WIDTH { PARAM_VALUE.FEATURE_COUNT_WIDTH } {
	# Procedure called to validate FEATURE_COUNT_WIDTH
	return true
}

proc update_PARAM_VALUE.FEATURE_WIDTH { PARAM_VALUE.FEATURE_WIDTH } {
	# Procedure called to update FEATURE_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.FEATURE_WIDTH { PARAM_VALUE.FEATURE_WIDTH } {
	# Procedure called to validate FEATURE_WIDTH
	return true
}

proc update_PARAM_VALUE.FEATURE_WORD_DEPTH { PARAM_VALUE.FEATURE_WORD_DEPTH } {
	# Procedure called to update FEATURE_WORD_DEPTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.FEATURE_WORD_DEPTH { PARAM_VALUE.FEATURE_WORD_DEPTH } {
	# Procedure called to validate FEATURE_WORD_DEPTH
	return true
}

proc update_PARAM_VALUE.FRAME_WIDTH { PARAM_VALUE.FRAME_WIDTH } {
	# Procedure called to update FRAME_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.FRAME_WIDTH { PARAM_VALUE.FRAME_WIDTH } {
	# Procedure called to validate FRAME_WIDTH
	return true
}

proc update_PARAM_VALUE.INTERNAL_MODULATION_BATCH { PARAM_VALUE.INTERNAL_MODULATION_BATCH } {
	# Procedure called to update INTERNAL_MODULATION_BATCH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.INTERNAL_MODULATION_BATCH { PARAM_VALUE.INTERNAL_MODULATION_BATCH } {
	# Procedure called to validate INTERNAL_MODULATION_BATCH
	return true
}

proc update_PARAM_VALUE.KERNEL_SIZE { PARAM_VALUE.KERNEL_SIZE } {
	# Procedure called to update KERNEL_SIZE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.KERNEL_SIZE { PARAM_VALUE.KERNEL_SIZE } {
	# Procedure called to validate KERNEL_SIZE
	return true
}

proc update_PARAM_VALUE.LANES { PARAM_VALUE.LANES } {
	# Procedure called to update LANES when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.LANES { PARAM_VALUE.LANES } {
	# Procedure called to validate LANES
	return true
}

proc update_PARAM_VALUE.MAX_BATCH { PARAM_VALUE.MAX_BATCH } {
	# Procedure called to update MAX_BATCH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.MAX_BATCH { PARAM_VALUE.MAX_BATCH } {
	# Procedure called to validate MAX_BATCH
	return true
}

proc update_PARAM_VALUE.MAX_CLASSES { PARAM_VALUE.MAX_CLASSES } {
	# Procedure called to update MAX_CLASSES when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.MAX_CLASSES { PARAM_VALUE.MAX_CLASSES } {
	# Procedure called to validate MAX_CLASSES
	return true
}

proc update_PARAM_VALUE.MAX_FEATURES { PARAM_VALUE.MAX_FEATURES } {
	# Procedure called to update MAX_FEATURES when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.MAX_FEATURES { PARAM_VALUE.MAX_FEATURES } {
	# Procedure called to validate MAX_FEATURES
	return true
}

proc update_PARAM_VALUE.MAX_INPUT_CHANNELS { PARAM_VALUE.MAX_INPUT_CHANNELS } {
	# Procedure called to update MAX_INPUT_CHANNELS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.MAX_INPUT_CHANNELS { PARAM_VALUE.MAX_INPUT_CHANNELS } {
	# Procedure called to validate MAX_INPUT_CHANNELS
	return true
}

proc update_PARAM_VALUE.MAX_OUTPUT_CHANNELS { PARAM_VALUE.MAX_OUTPUT_CHANNELS } {
	# Procedure called to update MAX_OUTPUT_CHANNELS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.MAX_OUTPUT_CHANNELS { PARAM_VALUE.MAX_OUTPUT_CHANNELS } {
	# Procedure called to validate MAX_OUTPUT_CHANNELS
	return true
}

proc update_PARAM_VALUE.MAX_TIMESTEPS { PARAM_VALUE.MAX_TIMESTEPS } {
	# Procedure called to update MAX_TIMESTEPS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.MAX_TIMESTEPS { PARAM_VALUE.MAX_TIMESTEPS } {
	# Procedure called to validate MAX_TIMESTEPS
	return true
}

proc update_PARAM_VALUE.MEMORY_ADDR_WIDTH { PARAM_VALUE.MEMORY_ADDR_WIDTH } {
	# Procedure called to update MEMORY_ADDR_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.MEMORY_ADDR_WIDTH { PARAM_VALUE.MEMORY_ADDR_WIDTH } {
	# Procedure called to validate MEMORY_ADDR_WIDTH
	return true
}

proc update_PARAM_VALUE.MEMORY_DEPTH { PARAM_VALUE.MEMORY_DEPTH } {
	# Procedure called to update MEMORY_DEPTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.MEMORY_DEPTH { PARAM_VALUE.MEMORY_DEPTH } {
	# Procedure called to validate MEMORY_DEPTH
	return true
}

proc update_PARAM_VALUE.OBS_INPUT_CHANNELS { PARAM_VALUE.OBS_INPUT_CHANNELS } {
	# Procedure called to update OBS_INPUT_CHANNELS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.OBS_INPUT_CHANNELS { PARAM_VALUE.OBS_INPUT_CHANNELS } {
	# Procedure called to validate OBS_INPUT_CHANNELS
	return true
}

proc update_PARAM_VALUE.OBS_OUTPUT_CHANNELS { PARAM_VALUE.OBS_OUTPUT_CHANNELS } {
	# Procedure called to update OBS_OUTPUT_CHANNELS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.OBS_OUTPUT_CHANNELS { PARAM_VALUE.OBS_OUTPUT_CHANNELS } {
	# Procedure called to validate OBS_OUTPUT_CHANNELS
	return true
}

proc update_PARAM_VALUE.PARAM_ADDR_WIDTH { PARAM_VALUE.PARAM_ADDR_WIDTH } {
	# Procedure called to update PARAM_ADDR_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.PARAM_ADDR_WIDTH { PARAM_VALUE.PARAM_ADDR_WIDTH } {
	# Procedure called to validate PARAM_ADDR_WIDTH
	return true
}

proc update_PARAM_VALUE.PARAM_DEPTH { PARAM_VALUE.PARAM_DEPTH } {
	# Procedure called to update PARAM_DEPTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.PARAM_DEPTH { PARAM_VALUE.PARAM_DEPTH } {
	# Procedure called to validate PARAM_DEPTH
	return true
}

proc update_PARAM_VALUE.POOLED_SPATIAL { PARAM_VALUE.POOLED_SPATIAL } {
	# Procedure called to update POOLED_SPATIAL when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.POOLED_SPATIAL { PARAM_VALUE.POOLED_SPATIAL } {
	# Procedure called to validate POOLED_SPATIAL
	return true
}

proc update_PARAM_VALUE.PROFILE_ID_WIDTH { PARAM_VALUE.PROFILE_ID_WIDTH } {
	# Procedure called to update PROFILE_ID_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.PROFILE_ID_WIDTH { PARAM_VALUE.PROFILE_ID_WIDTH } {
	# Procedure called to validate PROFILE_ID_WIDTH
	return true
}

proc update_PARAM_VALUE.Q16_ADDR_WIDTH { PARAM_VALUE.Q16_ADDR_WIDTH } {
	# Procedure called to update Q16_ADDR_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.Q16_ADDR_WIDTH { PARAM_VALUE.Q16_ADDR_WIDTH } {
	# Procedure called to validate Q16_ADDR_WIDTH
	return true
}

proc update_PARAM_VALUE.Q64_ADDR_WIDTH { PARAM_VALUE.Q64_ADDR_WIDTH } {
	# Procedure called to update Q64_ADDR_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.Q64_ADDR_WIDTH { PARAM_VALUE.Q64_ADDR_WIDTH } {
	# Procedure called to validate Q64_ADDR_WIDTH
	return true
}

proc update_PARAM_VALUE.RESIDUAL_WIDTH { PARAM_VALUE.RESIDUAL_WIDTH } {
	# Procedure called to update RESIDUAL_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.RESIDUAL_WIDTH { PARAM_VALUE.RESIDUAL_WIDTH } {
	# Procedure called to validate RESIDUAL_WIDTH
	return true
}

proc update_PARAM_VALUE.SAMPLE_WIDTH { PARAM_VALUE.SAMPLE_WIDTH } {
	# Procedure called to update SAMPLE_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SAMPLE_WIDTH { PARAM_VALUE.SAMPLE_WIDTH } {
	# Procedure called to validate SAMPLE_WIDTH
	return true
}

proc update_PARAM_VALUE.SCNN_WEIGHT_FRACTIONAL_BITS { PARAM_VALUE.SCNN_WEIGHT_FRACTIONAL_BITS } {
	# Procedure called to update SCNN_WEIGHT_FRACTIONAL_BITS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SCNN_WEIGHT_FRACTIONAL_BITS { PARAM_VALUE.SCNN_WEIGHT_FRACTIONAL_BITS } {
	# Procedure called to validate SCNN_WEIGHT_FRACTIONAL_BITS
	return true
}

proc update_PARAM_VALUE.SELECTED_LAYER { PARAM_VALUE.SELECTED_LAYER } {
	# Procedure called to update SELECTED_LAYER when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SELECTED_LAYER { PARAM_VALUE.SELECTED_LAYER } {
	# Procedure called to validate SELECTED_LAYER
	return true
}

proc update_PARAM_VALUE.SPATIAL { PARAM_VALUE.SPATIAL } {
	# Procedure called to update SPATIAL when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SPATIAL { PARAM_VALUE.SPATIAL } {
	# Procedure called to validate SPATIAL
	return true
}

proc update_PARAM_VALUE.SRNN_BACKBONE_MAX { PARAM_VALUE.SRNN_BACKBONE_MAX } {
	# Procedure called to update SRNN_BACKBONE_MAX when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SRNN_BACKBONE_MAX { PARAM_VALUE.SRNN_BACKBONE_MAX } {
	# Procedure called to validate SRNN_BACKBONE_MAX
	return true
}

proc update_PARAM_VALUE.SRNN_BACKBONE_PARAM_ADDR_WIDTH { PARAM_VALUE.SRNN_BACKBONE_PARAM_ADDR_WIDTH } {
	# Procedure called to update SRNN_BACKBONE_PARAM_ADDR_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SRNN_BACKBONE_PARAM_ADDR_WIDTH { PARAM_VALUE.SRNN_BACKBONE_PARAM_ADDR_WIDTH } {
	# Procedure called to validate SRNN_BACKBONE_PARAM_ADDR_WIDTH
	return true
}

proc update_PARAM_VALUE.SRNN_BACKBONE_PARAM_DEPTH { PARAM_VALUE.SRNN_BACKBONE_PARAM_DEPTH } {
	# Procedure called to update SRNN_BACKBONE_PARAM_DEPTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SRNN_BACKBONE_PARAM_DEPTH { PARAM_VALUE.SRNN_BACKBONE_PARAM_DEPTH } {
	# Procedure called to validate SRNN_BACKBONE_PARAM_DEPTH
	return true
}

proc update_PARAM_VALUE.SRNN_BATCH_MAX { PARAM_VALUE.SRNN_BATCH_MAX } {
	# Procedure called to update SRNN_BATCH_MAX when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SRNN_BATCH_MAX { PARAM_VALUE.SRNN_BATCH_MAX } {
	# Procedure called to validate SRNN_BATCH_MAX
	return true
}

proc update_PARAM_VALUE.SRNN_CLASS_MAX { PARAM_VALUE.SRNN_CLASS_MAX } {
	# Procedure called to update SRNN_CLASS_MAX when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SRNN_CLASS_MAX { PARAM_VALUE.SRNN_CLASS_MAX } {
	# Procedure called to validate SRNN_CLASS_MAX
	return true
}

proc update_PARAM_VALUE.SRNN_FEEDFORWARD_SOURCE_MAX { PARAM_VALUE.SRNN_FEEDFORWARD_SOURCE_MAX } {
	# Procedure called to update SRNN_FEEDFORWARD_SOURCE_MAX when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SRNN_FEEDFORWARD_SOURCE_MAX { PARAM_VALUE.SRNN_FEEDFORWARD_SOURCE_MAX } {
	# Procedure called to validate SRNN_FEEDFORWARD_SOURCE_MAX
	return true
}

proc update_PARAM_VALUE.SRNN_HIDDEN_MAX { PARAM_VALUE.SRNN_HIDDEN_MAX } {
	# Procedure called to update SRNN_HIDDEN_MAX when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SRNN_HIDDEN_MAX { PARAM_VALUE.SRNN_HIDDEN_MAX } {
	# Procedure called to validate SRNN_HIDDEN_MAX
	return true
}

proc update_PARAM_VALUE.SRNN_M2_BACKBONE_MAX { PARAM_VALUE.SRNN_M2_BACKBONE_MAX } {
	# Procedure called to update SRNN_M2_BACKBONE_MAX when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SRNN_M2_BACKBONE_MAX { PARAM_VALUE.SRNN_M2_BACKBONE_MAX } {
	# Procedure called to validate SRNN_M2_BACKBONE_MAX
	return true
}

proc update_PARAM_VALUE.SRNN_M2_BATCH_MAX { PARAM_VALUE.SRNN_M2_BATCH_MAX } {
	# Procedure called to update SRNN_M2_BATCH_MAX when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SRNN_M2_BATCH_MAX { PARAM_VALUE.SRNN_M2_BATCH_MAX } {
	# Procedure called to validate SRNN_M2_BATCH_MAX
	return true
}

proc update_PARAM_VALUE.SRNN_M2_CLASS_MAX { PARAM_VALUE.SRNN_M2_CLASS_MAX } {
	# Procedure called to update SRNN_M2_CLASS_MAX when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SRNN_M2_CLASS_MAX { PARAM_VALUE.SRNN_M2_CLASS_MAX } {
	# Procedure called to validate SRNN_M2_CLASS_MAX
	return true
}

proc update_PARAM_VALUE.SRNN_M2_HIDDEN_MAX { PARAM_VALUE.SRNN_M2_HIDDEN_MAX } {
	# Procedure called to update SRNN_M2_HIDDEN_MAX when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SRNN_M2_HIDDEN_MAX { PARAM_VALUE.SRNN_M2_HIDDEN_MAX } {
	# Procedure called to validate SRNN_M2_HIDDEN_MAX
	return true
}

proc update_PARAM_VALUE.SRNN_M2_PARAM_DEPTH { PARAM_VALUE.SRNN_M2_PARAM_DEPTH } {
	# Procedure called to update SRNN_M2_PARAM_DEPTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SRNN_M2_PARAM_DEPTH { PARAM_VALUE.SRNN_M2_PARAM_DEPTH } {
	# Procedure called to validate SRNN_M2_PARAM_DEPTH
	return true
}

proc update_PARAM_VALUE.SRNN_M2_TIMESTEP_MAX { PARAM_VALUE.SRNN_M2_TIMESTEP_MAX } {
	# Procedure called to update SRNN_M2_TIMESTEP_MAX when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SRNN_M2_TIMESTEP_MAX { PARAM_VALUE.SRNN_M2_TIMESTEP_MAX } {
	# Procedure called to validate SRNN_M2_TIMESTEP_MAX
	return true
}

proc update_PARAM_VALUE.SRNN_RECURRENT_SOURCE_MAX { PARAM_VALUE.SRNN_RECURRENT_SOURCE_MAX } {
	# Procedure called to update SRNN_RECURRENT_SOURCE_MAX when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SRNN_RECURRENT_SOURCE_MAX { PARAM_VALUE.SRNN_RECURRENT_SOURCE_MAX } {
	# Procedure called to validate SRNN_RECURRENT_SOURCE_MAX
	return true
}

proc update_PARAM_VALUE.SRNN_TIMESTEP_MAX { PARAM_VALUE.SRNN_TIMESTEP_MAX } {
	# Procedure called to update SRNN_TIMESTEP_MAX when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SRNN_TIMESTEP_MAX { PARAM_VALUE.SRNN_TIMESTEP_MAX } {
	# Procedure called to validate SRNN_TIMESTEP_MAX
	return true
}

proc update_PARAM_VALUE.TAG_WIDTH { PARAM_VALUE.TAG_WIDTH } {
	# Procedure called to update TAG_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.TAG_WIDTH { PARAM_VALUE.TAG_WIDTH } {
	# Procedure called to validate TAG_WIDTH
	return true
}

proc update_PARAM_VALUE.TENSOR_ID_WIDTH { PARAM_VALUE.TENSOR_ID_WIDTH } {
	# Procedure called to update TENSOR_ID_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.TENSOR_ID_WIDTH { PARAM_VALUE.TENSOR_ID_WIDTH } {
	# Procedure called to validate TENSOR_ID_WIDTH
	return true
}

proc update_PARAM_VALUE.TIMESTEP_COUNT_WIDTH { PARAM_VALUE.TIMESTEP_COUNT_WIDTH } {
	# Procedure called to update TIMESTEP_COUNT_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.TIMESTEP_COUNT_WIDTH { PARAM_VALUE.TIMESTEP_COUNT_WIDTH } {
	# Procedure called to validate TIMESTEP_COUNT_WIDTH
	return true
}

proc update_PARAM_VALUE.TIMESTEP_WIDTH { PARAM_VALUE.TIMESTEP_WIDTH } {
	# Procedure called to update TIMESTEP_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.TIMESTEP_WIDTH { PARAM_VALUE.TIMESTEP_WIDTH } {
	# Procedure called to validate TIMESTEP_WIDTH
	return true
}

proc update_PARAM_VALUE.TOPOLOGY_MASK { PARAM_VALUE.TOPOLOGY_MASK } {
	# Procedure called to update TOPOLOGY_MASK when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.TOPOLOGY_MASK { PARAM_VALUE.TOPOLOGY_MASK } {
	# Procedure called to validate TOPOLOGY_MASK
	return true
}

proc update_PARAM_VALUE.VERSION_WIDTH { PARAM_VALUE.VERSION_WIDTH } {
	# Procedure called to update VERSION_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.VERSION_WIDTH { PARAM_VALUE.VERSION_WIDTH } {
	# Procedure called to validate VERSION_WIDTH
	return true
}

proc update_PARAM_VALUE.WEIGHT_ADDR_WIDTH { PARAM_VALUE.WEIGHT_ADDR_WIDTH } {
	# Procedure called to update WEIGHT_ADDR_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.WEIGHT_ADDR_WIDTH { PARAM_VALUE.WEIGHT_ADDR_WIDTH } {
	# Procedure called to validate WEIGHT_ADDR_WIDTH
	return true
}

proc update_PARAM_VALUE.WEIGHT_DEPTH { PARAM_VALUE.WEIGHT_DEPTH } {
	# Procedure called to update WEIGHT_DEPTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.WEIGHT_DEPTH { PARAM_VALUE.WEIGHT_DEPTH } {
	# Procedure called to validate WEIGHT_DEPTH
	return true
}

proc update_PARAM_VALUE.WORK_ADDR_WIDTH { PARAM_VALUE.WORK_ADDR_WIDTH } {
	# Procedure called to update WORK_ADDR_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.WORK_ADDR_WIDTH { PARAM_VALUE.WORK_ADDR_WIDTH } {
	# Procedure called to validate WORK_ADDR_WIDTH
	return true
}


proc update_MODELPARAM_VALUE.TOPOLOGY_MASK { MODELPARAM_VALUE.TOPOLOGY_MASK PARAM_VALUE.TOPOLOGY_MASK } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.TOPOLOGY_MASK}] ${MODELPARAM_VALUE.TOPOLOGY_MASK}
}

proc update_MODELPARAM_VALUE.SCNN_WEIGHT_FRACTIONAL_BITS { MODELPARAM_VALUE.SCNN_WEIGHT_FRACTIONAL_BITS PARAM_VALUE.SCNN_WEIGHT_FRACTIONAL_BITS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SCNN_WEIGHT_FRACTIONAL_BITS}] ${MODELPARAM_VALUE.SCNN_WEIGHT_FRACTIONAL_BITS}
}

proc update_MODELPARAM_VALUE.ACCUMULATE_SCNN_SAMPLE_LOGITS { MODELPARAM_VALUE.ACCUMULATE_SCNN_SAMPLE_LOGITS PARAM_VALUE.ACCUMULATE_SCNN_SAMPLE_LOGITS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ACCUMULATE_SCNN_SAMPLE_LOGITS}] ${MODELPARAM_VALUE.ACCUMULATE_SCNN_SAMPLE_LOGITS}
}

proc update_MODELPARAM_VALUE.ENABLE_SCNN_FRAME_SERVICE { MODELPARAM_VALUE.ENABLE_SCNN_FRAME_SERVICE PARAM_VALUE.ENABLE_SCNN_FRAME_SERVICE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ENABLE_SCNN_FRAME_SERVICE}] ${MODELPARAM_VALUE.ENABLE_SCNN_FRAME_SERVICE}
}

proc update_MODELPARAM_VALUE.SRNN_M2_PARAM_DEPTH { MODELPARAM_VALUE.SRNN_M2_PARAM_DEPTH PARAM_VALUE.SRNN_M2_PARAM_DEPTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SRNN_M2_PARAM_DEPTH}] ${MODELPARAM_VALUE.SRNN_M2_PARAM_DEPTH}
}

proc update_MODELPARAM_VALUE.SRNN_M2_BATCH_MAX { MODELPARAM_VALUE.SRNN_M2_BATCH_MAX PARAM_VALUE.SRNN_M2_BATCH_MAX } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SRNN_M2_BATCH_MAX}] ${MODELPARAM_VALUE.SRNN_M2_BATCH_MAX}
}

proc update_MODELPARAM_VALUE.SRNN_M2_TIMESTEP_MAX { MODELPARAM_VALUE.SRNN_M2_TIMESTEP_MAX PARAM_VALUE.SRNN_M2_TIMESTEP_MAX } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SRNN_M2_TIMESTEP_MAX}] ${MODELPARAM_VALUE.SRNN_M2_TIMESTEP_MAX}
}

proc update_MODELPARAM_VALUE.SRNN_M2_BACKBONE_MAX { MODELPARAM_VALUE.SRNN_M2_BACKBONE_MAX PARAM_VALUE.SRNN_M2_BACKBONE_MAX } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SRNN_M2_BACKBONE_MAX}] ${MODELPARAM_VALUE.SRNN_M2_BACKBONE_MAX}
}

proc update_MODELPARAM_VALUE.SRNN_M2_HIDDEN_MAX { MODELPARAM_VALUE.SRNN_M2_HIDDEN_MAX PARAM_VALUE.SRNN_M2_HIDDEN_MAX } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SRNN_M2_HIDDEN_MAX}] ${MODELPARAM_VALUE.SRNN_M2_HIDDEN_MAX}
}

proc update_MODELPARAM_VALUE.SRNN_M2_CLASS_MAX { MODELPARAM_VALUE.SRNN_M2_CLASS_MAX PARAM_VALUE.SRNN_M2_CLASS_MAX } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SRNN_M2_CLASS_MAX}] ${MODELPARAM_VALUE.SRNN_M2_CLASS_MAX}
}

proc update_MODELPARAM_VALUE.SRNN_BACKBONE_PARAM_DEPTH { MODELPARAM_VALUE.SRNN_BACKBONE_PARAM_DEPTH PARAM_VALUE.SRNN_BACKBONE_PARAM_DEPTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SRNN_BACKBONE_PARAM_DEPTH}] ${MODELPARAM_VALUE.SRNN_BACKBONE_PARAM_DEPTH}
}

proc update_MODELPARAM_VALUE.SRNN_BACKBONE_PARAM_ADDR_WIDTH { MODELPARAM_VALUE.SRNN_BACKBONE_PARAM_ADDR_WIDTH PARAM_VALUE.SRNN_BACKBONE_PARAM_ADDR_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SRNN_BACKBONE_PARAM_ADDR_WIDTH}] ${MODELPARAM_VALUE.SRNN_BACKBONE_PARAM_ADDR_WIDTH}
}

proc update_MODELPARAM_VALUE.SRNN_FEEDFORWARD_SOURCE_MAX { MODELPARAM_VALUE.SRNN_FEEDFORWARD_SOURCE_MAX PARAM_VALUE.SRNN_FEEDFORWARD_SOURCE_MAX } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SRNN_FEEDFORWARD_SOURCE_MAX}] ${MODELPARAM_VALUE.SRNN_FEEDFORWARD_SOURCE_MAX}
}

proc update_MODELPARAM_VALUE.SRNN_RECURRENT_SOURCE_MAX { MODELPARAM_VALUE.SRNN_RECURRENT_SOURCE_MAX PARAM_VALUE.SRNN_RECURRENT_SOURCE_MAX } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SRNN_RECURRENT_SOURCE_MAX}] ${MODELPARAM_VALUE.SRNN_RECURRENT_SOURCE_MAX}
}

proc update_MODELPARAM_VALUE.WEIGHT_DEPTH { MODELPARAM_VALUE.WEIGHT_DEPTH PARAM_VALUE.WEIGHT_DEPTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.WEIGHT_DEPTH}] ${MODELPARAM_VALUE.WEIGHT_DEPTH}
}

proc update_MODELPARAM_VALUE.WEIGHT_ADDR_WIDTH { MODELPARAM_VALUE.WEIGHT_ADDR_WIDTH PARAM_VALUE.WEIGHT_ADDR_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.WEIGHT_ADDR_WIDTH}] ${MODELPARAM_VALUE.WEIGHT_ADDR_WIDTH}
}

proc update_MODELPARAM_VALUE.MEMORY_DEPTH { MODELPARAM_VALUE.MEMORY_DEPTH PARAM_VALUE.MEMORY_DEPTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.MEMORY_DEPTH}] ${MODELPARAM_VALUE.MEMORY_DEPTH}
}

proc update_MODELPARAM_VALUE.MEMORY_ADDR_WIDTH { MODELPARAM_VALUE.MEMORY_ADDR_WIDTH PARAM_VALUE.MEMORY_ADDR_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.MEMORY_ADDR_WIDTH}] ${MODELPARAM_VALUE.MEMORY_ADDR_WIDTH}
}

proc update_MODELPARAM_VALUE.WORK_ADDR_WIDTH { MODELPARAM_VALUE.WORK_ADDR_WIDTH PARAM_VALUE.WORK_ADDR_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.WORK_ADDR_WIDTH}] ${MODELPARAM_VALUE.WORK_ADDR_WIDTH}
}

proc update_MODELPARAM_VALUE.FEATURE_WORD_DEPTH { MODELPARAM_VALUE.FEATURE_WORD_DEPTH PARAM_VALUE.FEATURE_WORD_DEPTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.FEATURE_WORD_DEPTH}] ${MODELPARAM_VALUE.FEATURE_WORD_DEPTH}
}

proc update_MODELPARAM_VALUE.FEATURE_ADDR_WIDTH { MODELPARAM_VALUE.FEATURE_ADDR_WIDTH PARAM_VALUE.FEATURE_ADDR_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.FEATURE_ADDR_WIDTH}] ${MODELPARAM_VALUE.FEATURE_ADDR_WIDTH}
}

proc update_MODELPARAM_VALUE.PROFILE_ID_WIDTH { MODELPARAM_VALUE.PROFILE_ID_WIDTH PARAM_VALUE.PROFILE_ID_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.PROFILE_ID_WIDTH}] ${MODELPARAM_VALUE.PROFILE_ID_WIDTH}
}

proc update_MODELPARAM_VALUE.TENSOR_ID_WIDTH { MODELPARAM_VALUE.TENSOR_ID_WIDTH PARAM_VALUE.TENSOR_ID_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.TENSOR_ID_WIDTH}] ${MODELPARAM_VALUE.TENSOR_ID_WIDTH}
}

proc update_MODELPARAM_VALUE.PARAM_DEPTH { MODELPARAM_VALUE.PARAM_DEPTH PARAM_VALUE.PARAM_DEPTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.PARAM_DEPTH}] ${MODELPARAM_VALUE.PARAM_DEPTH}
}

proc update_MODELPARAM_VALUE.PARAM_ADDR_WIDTH { MODELPARAM_VALUE.PARAM_ADDR_WIDTH PARAM_VALUE.PARAM_ADDR_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.PARAM_ADDR_WIDTH}] ${MODELPARAM_VALUE.PARAM_ADDR_WIDTH}
}

proc update_MODELPARAM_VALUE.TAG_WIDTH { MODELPARAM_VALUE.TAG_WIDTH PARAM_VALUE.TAG_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.TAG_WIDTH}] ${MODELPARAM_VALUE.TAG_WIDTH}
}

proc update_MODELPARAM_VALUE.MAX_BATCH { MODELPARAM_VALUE.MAX_BATCH PARAM_VALUE.MAX_BATCH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.MAX_BATCH}] ${MODELPARAM_VALUE.MAX_BATCH}
}

proc update_MODELPARAM_VALUE.MAX_TIMESTEPS { MODELPARAM_VALUE.MAX_TIMESTEPS PARAM_VALUE.MAX_TIMESTEPS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.MAX_TIMESTEPS}] ${MODELPARAM_VALUE.MAX_TIMESTEPS}
}

proc update_MODELPARAM_VALUE.CLASSIFIER_CAPTURE_MAX_BATCH { MODELPARAM_VALUE.CLASSIFIER_CAPTURE_MAX_BATCH PARAM_VALUE.CLASSIFIER_CAPTURE_MAX_BATCH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CLASSIFIER_CAPTURE_MAX_BATCH}] ${MODELPARAM_VALUE.CLASSIFIER_CAPTURE_MAX_BATCH}
}

proc update_MODELPARAM_VALUE.CLASSIFIER_CAPTURE_MAX_TIMESTEPS { MODELPARAM_VALUE.CLASSIFIER_CAPTURE_MAX_TIMESTEPS PARAM_VALUE.CLASSIFIER_CAPTURE_MAX_TIMESTEPS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CLASSIFIER_CAPTURE_MAX_TIMESTEPS}] ${MODELPARAM_VALUE.CLASSIFIER_CAPTURE_MAX_TIMESTEPS}
}

proc update_MODELPARAM_VALUE.MAX_INPUT_CHANNELS { MODELPARAM_VALUE.MAX_INPUT_CHANNELS PARAM_VALUE.MAX_INPUT_CHANNELS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.MAX_INPUT_CHANNELS}] ${MODELPARAM_VALUE.MAX_INPUT_CHANNELS}
}

proc update_MODELPARAM_VALUE.MAX_OUTPUT_CHANNELS { MODELPARAM_VALUE.MAX_OUTPUT_CHANNELS PARAM_VALUE.MAX_OUTPUT_CHANNELS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.MAX_OUTPUT_CHANNELS}] ${MODELPARAM_VALUE.MAX_OUTPUT_CHANNELS}
}

proc update_MODELPARAM_VALUE.OBS_INPUT_CHANNELS { MODELPARAM_VALUE.OBS_INPUT_CHANNELS PARAM_VALUE.OBS_INPUT_CHANNELS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.OBS_INPUT_CHANNELS}] ${MODELPARAM_VALUE.OBS_INPUT_CHANNELS}
}

proc update_MODELPARAM_VALUE.OBS_OUTPUT_CHANNELS { MODELPARAM_VALUE.OBS_OUTPUT_CHANNELS PARAM_VALUE.OBS_OUTPUT_CHANNELS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.OBS_OUTPUT_CHANNELS}] ${MODELPARAM_VALUE.OBS_OUTPUT_CHANNELS}
}

proc update_MODELPARAM_VALUE.SPATIAL { MODELPARAM_VALUE.SPATIAL PARAM_VALUE.SPATIAL } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SPATIAL}] ${MODELPARAM_VALUE.SPATIAL}
}

proc update_MODELPARAM_VALUE.POOLED_SPATIAL { MODELPARAM_VALUE.POOLED_SPATIAL PARAM_VALUE.POOLED_SPATIAL } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.POOLED_SPATIAL}] ${MODELPARAM_VALUE.POOLED_SPATIAL}
}

proc update_MODELPARAM_VALUE.KERNEL_SIZE { MODELPARAM_VALUE.KERNEL_SIZE PARAM_VALUE.KERNEL_SIZE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.KERNEL_SIZE}] ${MODELPARAM_VALUE.KERNEL_SIZE}
}

proc update_MODELPARAM_VALUE.Q16_ADDR_WIDTH { MODELPARAM_VALUE.Q16_ADDR_WIDTH PARAM_VALUE.Q16_ADDR_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.Q16_ADDR_WIDTH}] ${MODELPARAM_VALUE.Q16_ADDR_WIDTH}
}

proc update_MODELPARAM_VALUE.Q64_ADDR_WIDTH { MODELPARAM_VALUE.Q64_ADDR_WIDTH PARAM_VALUE.Q64_ADDR_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.Q64_ADDR_WIDTH}] ${MODELPARAM_VALUE.Q64_ADDR_WIDTH}
}

proc update_MODELPARAM_VALUE.VERSION_WIDTH { MODELPARAM_VALUE.VERSION_WIDTH PARAM_VALUE.VERSION_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.VERSION_WIDTH}] ${MODELPARAM_VALUE.VERSION_WIDTH}
}

proc update_MODELPARAM_VALUE.MAX_FEATURES { MODELPARAM_VALUE.MAX_FEATURES PARAM_VALUE.MAX_FEATURES } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.MAX_FEATURES}] ${MODELPARAM_VALUE.MAX_FEATURES}
}

proc update_MODELPARAM_VALUE.MAX_CLASSES { MODELPARAM_VALUE.MAX_CLASSES PARAM_VALUE.MAX_CLASSES } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.MAX_CLASSES}] ${MODELPARAM_VALUE.MAX_CLASSES}
}

proc update_MODELPARAM_VALUE.LANES { MODELPARAM_VALUE.LANES PARAM_VALUE.LANES } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.LANES}] ${MODELPARAM_VALUE.LANES}
}

proc update_MODELPARAM_VALUE.ENGINES { MODELPARAM_VALUE.ENGINES PARAM_VALUE.ENGINES } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ENGINES}] ${MODELPARAM_VALUE.ENGINES}
}

proc update_MODELPARAM_VALUE.FRAME_WIDTH { MODELPARAM_VALUE.FRAME_WIDTH PARAM_VALUE.FRAME_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.FRAME_WIDTH}] ${MODELPARAM_VALUE.FRAME_WIDTH}
}

proc update_MODELPARAM_VALUE.SELECTED_LAYER { MODELPARAM_VALUE.SELECTED_LAYER PARAM_VALUE.SELECTED_LAYER } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SELECTED_LAYER}] ${MODELPARAM_VALUE.SELECTED_LAYER}
}

proc update_MODELPARAM_VALUE.ENABLE_SCNN_CONV8_LEARNING { MODELPARAM_VALUE.ENABLE_SCNN_CONV8_LEARNING PARAM_VALUE.ENABLE_SCNN_CONV8_LEARNING } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ENABLE_SCNN_CONV8_LEARNING}] ${MODELPARAM_VALUE.ENABLE_SCNN_CONV8_LEARNING}
}

proc update_MODELPARAM_VALUE.INTERNAL_MODULATION_BATCH { MODELPARAM_VALUE.INTERNAL_MODULATION_BATCH PARAM_VALUE.INTERNAL_MODULATION_BATCH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.INTERNAL_MODULATION_BATCH}] ${MODELPARAM_VALUE.INTERNAL_MODULATION_BATCH}
}

proc update_MODELPARAM_VALUE.SAMPLE_WIDTH { MODELPARAM_VALUE.SAMPLE_WIDTH PARAM_VALUE.SAMPLE_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SAMPLE_WIDTH}] ${MODELPARAM_VALUE.SAMPLE_WIDTH}
}

proc update_MODELPARAM_VALUE.BATCH_COUNT_WIDTH { MODELPARAM_VALUE.BATCH_COUNT_WIDTH PARAM_VALUE.BATCH_COUNT_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BATCH_COUNT_WIDTH}] ${MODELPARAM_VALUE.BATCH_COUNT_WIDTH}
}

proc update_MODELPARAM_VALUE.TIMESTEP_WIDTH { MODELPARAM_VALUE.TIMESTEP_WIDTH PARAM_VALUE.TIMESTEP_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.TIMESTEP_WIDTH}] ${MODELPARAM_VALUE.TIMESTEP_WIDTH}
}

proc update_MODELPARAM_VALUE.TIMESTEP_COUNT_WIDTH { MODELPARAM_VALUE.TIMESTEP_COUNT_WIDTH PARAM_VALUE.TIMESTEP_COUNT_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.TIMESTEP_COUNT_WIDTH}] ${MODELPARAM_VALUE.TIMESTEP_COUNT_WIDTH}
}

proc update_MODELPARAM_VALUE.CHANNEL_WIDTH { MODELPARAM_VALUE.CHANNEL_WIDTH PARAM_VALUE.CHANNEL_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CHANNEL_WIDTH}] ${MODELPARAM_VALUE.CHANNEL_WIDTH}
}

proc update_MODELPARAM_VALUE.CHANNEL_COUNT_WIDTH { MODELPARAM_VALUE.CHANNEL_COUNT_WIDTH PARAM_VALUE.CHANNEL_COUNT_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CHANNEL_COUNT_WIDTH}] ${MODELPARAM_VALUE.CHANNEL_COUNT_WIDTH}
}

proc update_MODELPARAM_VALUE.FEATURE_WIDTH { MODELPARAM_VALUE.FEATURE_WIDTH PARAM_VALUE.FEATURE_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.FEATURE_WIDTH}] ${MODELPARAM_VALUE.FEATURE_WIDTH}
}

proc update_MODELPARAM_VALUE.FEATURE_COUNT_WIDTH { MODELPARAM_VALUE.FEATURE_COUNT_WIDTH PARAM_VALUE.FEATURE_COUNT_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.FEATURE_COUNT_WIDTH}] ${MODELPARAM_VALUE.FEATURE_COUNT_WIDTH}
}

proc update_MODELPARAM_VALUE.CLASS_WIDTH { MODELPARAM_VALUE.CLASS_WIDTH PARAM_VALUE.CLASS_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CLASS_WIDTH}] ${MODELPARAM_VALUE.CLASS_WIDTH}
}

proc update_MODELPARAM_VALUE.CLASS_COUNT_WIDTH { MODELPARAM_VALUE.CLASS_COUNT_WIDTH PARAM_VALUE.CLASS_COUNT_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CLASS_COUNT_WIDTH}] ${MODELPARAM_VALUE.CLASS_COUNT_WIDTH}
}

proc update_MODELPARAM_VALUE.ENABLE_ONLINE_LEARNING { MODELPARAM_VALUE.ENABLE_ONLINE_LEARNING PARAM_VALUE.ENABLE_ONLINE_LEARNING } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ENABLE_ONLINE_LEARNING}] ${MODELPARAM_VALUE.ENABLE_ONLINE_LEARNING}
}

proc update_MODELPARAM_VALUE.ENABLE_SRNN_WH1_LEARNING { MODELPARAM_VALUE.ENABLE_SRNN_WH1_LEARNING PARAM_VALUE.ENABLE_SRNN_WH1_LEARNING } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ENABLE_SRNN_WH1_LEARNING}] ${MODELPARAM_VALUE.ENABLE_SRNN_WH1_LEARNING}
}

proc update_MODELPARAM_VALUE.CODE_WIDTH { MODELPARAM_VALUE.CODE_WIDTH PARAM_VALUE.CODE_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CODE_WIDTH}] ${MODELPARAM_VALUE.CODE_WIDTH}
}

proc update_MODELPARAM_VALUE.RESIDUAL_WIDTH { MODELPARAM_VALUE.RESIDUAL_WIDTH PARAM_VALUE.RESIDUAL_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.RESIDUAL_WIDTH}] ${MODELPARAM_VALUE.RESIDUAL_WIDTH}
}

proc update_MODELPARAM_VALUE.FC_MAX_HIDDEN { MODELPARAM_VALUE.FC_MAX_HIDDEN PARAM_VALUE.FC_MAX_HIDDEN } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.FC_MAX_HIDDEN}] ${MODELPARAM_VALUE.FC_MAX_HIDDEN}
}

proc update_MODELPARAM_VALUE.FC_MAX_OUTPUTS { MODELPARAM_VALUE.FC_MAX_OUTPUTS PARAM_VALUE.FC_MAX_OUTPUTS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.FC_MAX_OUTPUTS}] ${MODELPARAM_VALUE.FC_MAX_OUTPUTS}
}

proc update_MODELPARAM_VALUE.SRNN_BATCH_MAX { MODELPARAM_VALUE.SRNN_BATCH_MAX PARAM_VALUE.SRNN_BATCH_MAX } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SRNN_BATCH_MAX}] ${MODELPARAM_VALUE.SRNN_BATCH_MAX}
}

proc update_MODELPARAM_VALUE.SRNN_TIMESTEP_MAX { MODELPARAM_VALUE.SRNN_TIMESTEP_MAX PARAM_VALUE.SRNN_TIMESTEP_MAX } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SRNN_TIMESTEP_MAX}] ${MODELPARAM_VALUE.SRNN_TIMESTEP_MAX}
}

proc update_MODELPARAM_VALUE.SRNN_HIDDEN_MAX { MODELPARAM_VALUE.SRNN_HIDDEN_MAX PARAM_VALUE.SRNN_HIDDEN_MAX } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SRNN_HIDDEN_MAX}] ${MODELPARAM_VALUE.SRNN_HIDDEN_MAX}
}

proc update_MODELPARAM_VALUE.SRNN_BACKBONE_MAX { MODELPARAM_VALUE.SRNN_BACKBONE_MAX PARAM_VALUE.SRNN_BACKBONE_MAX } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SRNN_BACKBONE_MAX}] ${MODELPARAM_VALUE.SRNN_BACKBONE_MAX}
}

proc update_MODELPARAM_VALUE.SRNN_CLASS_MAX { MODELPARAM_VALUE.SRNN_CLASS_MAX PARAM_VALUE.SRNN_CLASS_MAX } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SRNN_CLASS_MAX}] ${MODELPARAM_VALUE.SRNN_CLASS_MAX}
}

