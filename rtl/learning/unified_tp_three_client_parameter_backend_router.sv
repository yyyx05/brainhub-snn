`timescale 1ns/1ps
`default_nettype none

// Topology-neutral backend router for FC, SRNN and SCNN (client indices 0..2,
// owner encodings 1..3). Response ownership is retained until handshake, so a
// profile switch can use o_quiescent as its release condition.
module unified_tp_three_client_parameter_backend_router #(
    parameter int CLIENTS=3, PARAM_ADDR_WIDTH=24, TAG_WIDTH=16,
    parameter int CODE_WIDTH=16, RESIDUAL_WIDTH=16
) (
    input logic i_clk,i_rst_n,i_clear_error,i_owner_active,
    input logic [1:0] i_active_owner,
    input logic [CLIENTS-1:0] i_load_valid,
    output logic [CLIENTS-1:0] o_load_ready,
    input logic [CLIENTS*PARAM_ADDR_WIDTH-1:0] i_load_addr,
    input logic signed [CLIENTS*CODE_WIDTH-1:0] i_load_code,
    input logic signed [CLIENTS*RESIDUAL_WIDTH-1:0] i_load_residual,
    input logic [CLIENTS-1:0] i_read_valid,
    output logic [CLIENTS-1:0] o_read_ready,
    input logic [CLIENTS*PARAM_ADDR_WIDTH-1:0] i_read_addr,
    input logic [CLIENTS*TAG_WIDTH-1:0] i_read_tag,
    output logic [CLIENTS-1:0] o_response_valid,
    input logic [CLIENTS-1:0] i_response_ready,
    output logic [CLIENTS*TAG_WIDTH-1:0] o_response_tag,
    output logic signed [CLIENTS*CODE_WIDTH-1:0] o_response_code,
    output logic signed [CLIENTS*RESIDUAL_WIDTH-1:0] o_response_residual,
    output logic [CLIENTS-1:0] o_response_error,
    input logic [CLIENTS-1:0] i_write_valid,
    output logic [CLIENTS-1:0] o_write_ready,
    input logic [CLIENTS*PARAM_ADDR_WIDTH-1:0] i_write_addr,
    input logic signed [CLIENTS*CODE_WIDTH-1:0] i_write_code,
    input logic signed [CLIENTS*RESIDUAL_WIDTH-1:0] i_write_residual,
    input logic [CLIENTS-1:0] i_write_code_enable,i_write_residual_enable,
    input logic [CLIENTS-1:0] i_forward_read_valid,
    output logic [CLIENTS-1:0] o_forward_read_ready,
    input logic [CLIENTS*PARAM_ADDR_WIDTH-1:0] i_forward_read_addr,
    input logic [CLIENTS*TAG_WIDTH-1:0] i_forward_read_tag,
    output logic [CLIENTS-1:0] o_forward_response_valid,
    input logic [CLIENTS-1:0] i_forward_response_ready,
    output logic [CLIENTS*TAG_WIDTH-1:0] o_forward_response_tag,
    output logic signed [CLIENTS*CODE_WIDTH-1:0] o_forward_response_code,
    output logic [CLIENTS-1:0] o_forward_response_error,

    output logic o_store_load_valid,input logic i_store_load_ready,
    output logic [PARAM_ADDR_WIDTH-1:0] o_store_load_addr,
    output logic signed [CODE_WIDTH-1:0] o_store_load_code,
    output logic signed [RESIDUAL_WIDTH-1:0] o_store_load_residual,
    output logic o_store_learning_read_valid,input logic i_store_learning_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0] o_store_learning_read_addr,
    output logic [TAG_WIDTH-1:0] o_store_learning_read_tag,
    input logic i_store_learning_response_valid,output logic o_store_learning_response_ready,
    input logic [TAG_WIDTH-1:0] i_store_learning_response_tag,
    input logic signed [CODE_WIDTH-1:0] i_store_learning_response_code,
    input logic signed [RESIDUAL_WIDTH-1:0] i_store_learning_response_residual,
    input logic i_store_learning_response_error,
    output logic o_store_learning_write_valid,input logic i_store_learning_write_ready,
    output logic [PARAM_ADDR_WIDTH-1:0] o_store_learning_write_addr,
    output logic signed [CODE_WIDTH-1:0] o_store_learning_write_code,
    output logic signed [RESIDUAL_WIDTH-1:0] o_store_learning_write_residual,
    output logic o_store_learning_write_code_enable,o_store_learning_write_residual_enable,
    output logic o_store_forward_read_valid,input logic i_store_forward_read_ready,
    output logic [PARAM_ADDR_WIDTH-1:0] o_store_forward_read_addr,
    output logic [TAG_WIDTH-1:0] o_store_forward_read_tag,
    input logic i_store_forward_response_valid,output logic o_store_forward_response_ready,
    input logic [TAG_WIDTH-1:0] i_store_forward_response_tag,
    input logic signed [CODE_WIDTH-1:0] i_store_forward_response_code,
    input logic i_store_forward_response_error,input logic i_store_protocol_error,
    output logic o_quiescent,o_protocol_error
);
    logic learning_pending_q,forward_pending_q;
    logic [1:0] learning_owner_q,forward_owner_q;
    logic [CLIENTS-1:0] active_onehot,all_requests;
    logic learning_read_fire,learning_response_fire,forward_read_fire,forward_response_fire;
    logic [1:0] active_index;

    always_comb begin
        active_onehot = '0;
        active_index = '0;
        if (i_owner_active) begin
            unique case (i_active_owner)
                2'd1: begin active_onehot[0] = 1'b1; active_index = 2'd0; end
                2'd2: begin active_onehot[1] = 1'b1; active_index = 2'd1; end
                2'd3: begin active_onehot[2] = 1'b1; active_index = 2'd2; end
                default: begin active_onehot = '0; active_index = '0; end
            endcase
        end
    end

    // Request routing is independent of the response path. Keeping it in a
    // separate block avoids introducing a false ready/valid combinational loop
    // when the backing store derives load readiness from its request inputs.
    always_comb begin
        o_load_ready='0;o_read_ready='0;o_write_ready='0;o_forward_read_ready='0;
        o_store_load_valid=1'b0;o_store_load_addr='0;o_store_load_code='0;o_store_load_residual='0;
        o_store_learning_read_valid=1'b0;o_store_learning_read_addr='0;o_store_learning_read_tag='0;
        o_store_learning_write_valid=1'b0;o_store_learning_write_addr='0;o_store_learning_write_code='0;
        o_store_learning_write_residual='0;o_store_learning_write_code_enable=1'b0;
        o_store_learning_write_residual_enable=1'b0;
        o_store_forward_read_valid=1'b0;o_store_forward_read_addr='0;o_store_forward_read_tag='0;
        if(|active_onehot) begin
            o_store_load_valid=i_load_valid[active_index];o_load_ready[active_index]=i_store_load_ready;
            o_store_load_addr=i_load_addr[active_index*PARAM_ADDR_WIDTH+:PARAM_ADDR_WIDTH];
            o_store_load_code=i_load_code[active_index*CODE_WIDTH+:CODE_WIDTH];
            o_store_load_residual=i_load_residual[active_index*RESIDUAL_WIDTH+:RESIDUAL_WIDTH];
            // The router retains one response owner, so it must not accept a
            // second request until that response has been consumed.
            o_store_learning_read_valid=i_read_valid[active_index]&&!learning_pending_q;
            o_read_ready[active_index]=i_store_learning_read_ready&&!learning_pending_q;
            o_store_learning_read_addr=i_read_addr[active_index*PARAM_ADDR_WIDTH+:PARAM_ADDR_WIDTH];
            o_store_learning_read_tag=i_read_tag[active_index*TAG_WIDTH+:TAG_WIDTH];
            o_store_learning_write_valid=i_write_valid[active_index];o_write_ready[active_index]=i_store_learning_write_ready;
            o_store_learning_write_addr=i_write_addr[active_index*PARAM_ADDR_WIDTH+:PARAM_ADDR_WIDTH];
            o_store_learning_write_code=i_write_code[active_index*CODE_WIDTH+:CODE_WIDTH];
            o_store_learning_write_residual=i_write_residual[active_index*RESIDUAL_WIDTH+:RESIDUAL_WIDTH];
            o_store_learning_write_code_enable=i_write_code_enable[active_index];
            o_store_learning_write_residual_enable=i_write_residual_enable[active_index];
            o_store_forward_read_valid=i_forward_read_valid[active_index]&&!forward_pending_q;
            o_forward_read_ready[active_index]=i_store_forward_read_ready&&!forward_pending_q;
            o_store_forward_read_addr=i_forward_read_addr[active_index*PARAM_ADDR_WIDTH+:PARAM_ADDR_WIDTH];
            o_store_forward_read_tag=i_forward_read_tag[active_index*TAG_WIDTH+:TAG_WIDTH];
        end
    end

    always_comb begin
        o_response_valid='0;o_forward_response_valid='0;
        o_response_tag={CLIENTS{i_store_learning_response_tag}};
        o_response_code={CLIENTS{i_store_learning_response_code}};
        o_response_residual={CLIENTS{i_store_learning_response_residual}};
        o_response_error={CLIENTS{i_store_learning_response_error}};
        o_forward_response_tag={CLIENTS{i_store_forward_response_tag}};
        o_forward_response_code={CLIENTS{i_store_forward_response_code}};
        o_forward_response_error={CLIENTS{i_store_forward_response_error}};
        if(learning_pending_q) begin
            o_response_valid[learning_owner_q-1'b1]=i_store_learning_response_valid;
            o_store_learning_response_ready=i_response_ready[learning_owner_q-1'b1];
        end else o_store_learning_response_ready=1'b0;
        if(forward_pending_q) begin
            o_forward_response_valid[forward_owner_q-1'b1]=i_store_forward_response_valid;
            o_store_forward_response_ready=i_forward_response_ready[forward_owner_q-1'b1];
        end else o_store_forward_response_ready=1'b0;
    end

    assign learning_read_fire=o_store_learning_read_valid&&i_store_learning_read_ready;
    assign learning_response_fire=i_store_learning_response_valid&&o_store_learning_response_ready;
    assign forward_read_fire=o_store_forward_read_valid&&i_store_forward_read_ready;
    assign forward_response_fire=i_store_forward_response_valid&&o_store_forward_response_ready;
    assign all_requests=i_load_valid|i_read_valid|i_write_valid|i_forward_read_valid;
    assign o_quiescent=!learning_pending_q&&!forward_pending_q&&!i_store_learning_response_valid&&
        !i_store_forward_response_valid&&!o_store_load_valid&&!o_store_learning_read_valid&&
        !o_store_learning_write_valid&&!o_store_forward_read_valid;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if(!i_rst_n) begin
            learning_pending_q<=0;forward_pending_q<=0;learning_owner_q<=0;forward_owner_q<=0;o_protocol_error<=0;
        end else begin
            if(i_clear_error)o_protocol_error<=0;
            if(i_store_protocol_error|(|(all_requests&~active_onehot))|
               (learning_pending_q&&(!i_owner_active||i_active_owner!=learning_owner_q))|
               (forward_pending_q&&(!i_owner_active||i_active_owner!=forward_owner_q))|
               (i_store_learning_response_valid&&!learning_pending_q)|
               (i_store_forward_response_valid&&!forward_pending_q))o_protocol_error<=1;
            if(learning_response_fire&&!learning_read_fire)begin learning_pending_q<=0;learning_owner_q<=0;end
            if(learning_read_fire)begin learning_pending_q<=1;learning_owner_q<=i_active_owner;end
            if(forward_response_fire&&!forward_read_fire)begin forward_pending_q<=0;forward_owner_q<=0;end
            if(forward_read_fire)begin forward_pending_q<=1;forward_owner_q<=i_active_owner;end
        end
    end
endmodule

`default_nettype wire
