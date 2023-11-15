def copy_weights(self, model):
  ## Embedding
  self.embedding.weight.data = model.model.shared.weight.data

  ## Positional encoding
  self.positional_encoding_src.weight.data = \
    model.model.encoder.embed_positions.weight.data
  self.positional_encoding_tgt.weight.data = \
    model.model.decoder.embed_positions.weight.data

  assert len(self.encoder) == len(model.model.encoder.layers)    
  ## Encoder
  for i in range(len(model.model.encoder.layers)):
    ## Self-attention
    self.encoder[i].self_attn.attn.k_proj.weight.data = \
      model.model.encoder.layers[i].self_attn.k_proj.weight.data
    self.encoder[i].self_attn.attn.k_proj.bias.data = \
      model.model.encoder.layers[i].self_attn.k_proj.bias.data
    self.encoder[i].self_attn.attn.v_proj.weight.data = \
      model.model.encoder.layers[i].self_attn.v_proj.weight.data
    self.encoder[i].self_attn.attn.v_proj.bias.data = \
      model.model.encoder.layers[i].self_attn.v_proj.bias.data
    self.encoder[i].self_attn.attn.q_proj.weight.data = \
      model.model.encoder.layers[i].self_attn.q_proj.weight.data
    self.encoder[i].self_attn.attn.q_proj.bias.data = \
      model.model.encoder.layers[i].self_attn.q_proj.bias.data
    self.encoder[i].self_attn.attn.out_proj.weight.data = \
      model.model.encoder.layers[i].self_attn.out_proj.weight.data
    self.encoder[i].self_attn.attn.out_proj.bias.data = \
      model.model.encoder.layers[i].self_attn.out_proj.bias.data

    ## MLP
    self.encoder[i].mlp.fc1.weight.data = \
      model.model.encoder.layers[i].fc1.weight.data
    self.encoder[i].mlp.fc1.bias.data = \
      model.model.encoder.layers[i].fc1.bias.data
    self.encoder[i].mlp.fc2.weight.data = \
      model.model.encoder.layers[i].fc2.weight.data
    self.encoder[i].mlp.fc2.bias.data = \
      model.model.encoder.layers[i].fc2.bias.data

    ## Layer normalization
    self.encoder[i].self_attn_layer_norm.weight.data = \
      model.model.encoder.layers[i].self_attn_layer_norm.weight.data
    self.encoder[i].self_attn_layer_norm.bias.data = \
      model.model.encoder.layers[i].self_attn_layer_norm.bias.data
    self.encoder[i].final_layer_norm.weight.data = \
      model.model.encoder.layers[i].final_layer_norm.weight.data
    self.encoder[i].final_layer_norm.bias.data = \
      model.model.encoder.layers[i].final_layer_norm.bias.data

  assert len(self.decoder) == len(model.model.decoder.layers)
  ## Decoder
  for i in range(len(model.model.decoder.layers)):
    ## Self-attention
    self.decoder[i].self_attn.attn.k_proj.weight.data = \
      model.model.decoder.layers[i].self_attn.k_proj.weight.data
    self.decoder[i].self_attn.attn.k_proj.bias.data = \
      model.model.decoder.layers[i].self_attn.k_proj.bias.data
    self.decoder[i].self_attn.attn.v_proj.weight.data = \
      model.model.decoder.layers[i].self_attn.v_proj.weight.data
    self.decoder[i].self_attn.attn.v_proj.bias.data = \
      model.model.decoder.layers[i].self_attn.v_proj.bias.data
    self.decoder[i].self_attn.attn.q_proj.weight.data = \
      model.model.decoder.layers[i].self_attn.q_proj.weight.data
    self.decoder[i].self_attn.attn.q_proj.bias.data = \
      model.model.decoder.layers[i].self_attn.q_proj.bias.data
    self.decoder[i].self_attn.attn.out_proj.weight.data = \
      model.model.decoder.layers[i].self_attn.out_proj.weight.data
    self.decoder[i].self_attn.attn.out_proj.bias.data = \
      model.model.decoder.layers[i].self_attn.out_proj.bias.data

    ## Cross-attention
    self.decoder[i].cross_attn.k_proj.weight.data = \
      model.model.decoder.layers[i].encoder_attn.k_proj.weight.data
    self.decoder[i].cross_attn.k_proj.bias.data = \
      model.model.decoder.layers[i].encoder_attn.k_proj.bias.data
    self.decoder[i].cross_attn.v_proj.weight.data = \
      model.model.decoder.layers[i].encoder_attn.v_proj.weight.data
    self.decoder[i].cross_attn.v_proj.bias.data = \
      model.model.decoder.layers[i].encoder_attn.v_proj.bias.data
    self.decoder[i].cross_attn.q_proj.weight.data = \
      model.model.decoder.layers[i].encoder_attn.q_proj.weight.data
    self.decoder[i].cross_attn.q_proj.bias.data = \
      model.model.decoder.layers[i].encoder_attn.q_proj.bias.data
    self.decoder[i].cross_attn.out_proj.weight.data = \
      model.model.decoder.layers[i].encoder_attn.out_proj.weight.data
    self.decoder[i].cross_attn.out_proj.bias.data = \
      model.model.decoder.layers[i].encoder_attn.out_proj.bias.data      

    ## MLP
    self.decoder[i].mlp.fc1.weight.data = \
      model.model.decoder.layers[i].fc1.weight.data
    self.decoder[i].mlp.fc1.bias.data = \
      model.model.decoder.layers[i].fc1.bias.data
    self.decoder[i].mlp.fc2.weight.data = \
      model.model.decoder.layers[i].fc2.weight.data
    self.decoder[i].mlp.fc2.bias.data = \
      model.model.decoder.layers[i].fc2.bias.data

    ## Layer normalization
    self.decoder[i].self_attn_layer_norm.weight.data = \
      model.model.decoder.layers[i].self_attn_layer_norm.weight.data
    self.decoder[i].self_attn_layer_norm.bias.data = \
      model.model.decoder.layers[i].self_attn_layer_norm.bias.data
    self.decoder[i].cross_attn_layer_norm.weight.data = \
      model.model.decoder.layers[i].encoder_attn_layer_norm.weight.data
    self.decoder[i].cross_attn_layer_norm.bias.data = \
      model.model.decoder.layers[i].encoder_attn_layer_norm.bias.data
    self.decoder[i].final_layer_norm.weight.data = \
      model.model.decoder.layers[i].final_layer_norm.weight.data
    self.decoder[i].final_layer_norm.bias.data = \
      model.model.decoder.layers[i].final_layer_norm.bias.data

  ## Classifier
  self.classifier.weight.data = model.lm_head.weight.data
  self.classifier.bias.data = model.final_logits_bias.reshape(
    self.classifier.bias.data.shape
  )

def sequential(self, x, *args, **kwargs):  
  for layer in self:
    x = layer(x, *args, **kwargs)

  return x






















