# Simple Charged Particle Tracking Edge Classification GNN hls4ml Implementation

## Generating weights and input graphs
- Assuming trained model in: https://github.com/jmduarte/exatrkx-neurips19/tree/master/gnn-tracking
- Run this script: https://github.com/jmduarte/exatrkx-neurips19/blob/master/gnn-tracking/scripts/evaluate_gnn_simple 
```
python scripts/evaluate_gnn_simple configs/train_edge_classifier_kaggle_share.yaml  --iteration -1 --ckpt out/segments_1_2GeV_8dim/v0_kaggle/
```
- Then copy weights and any single set of graph data to the hls4ml project directory:
```
cp tb_data/tb_input_edge_features_00000.dat /path_to_hls4ml/example-prjs/graph/gnn_simple/tb_input_edge_features.dat
cp tb_data/tb_input_node_features_00000.dat /path_to_hls4ml/example-prjs/graph/gnn_simple/tb_input_node_features.dat
cp tb_data/tb_output_edge_predictions_00000.dat /path_to_hls4ml/example-prjs/graph/gnn_simple/tb_output_edge_predictions.dat
cp tb_data/tb_receivers_00000.dat /path_to_hls4ml/example-prjs/graph/gnn_simple/tb_receivers.dat
cp tb_data/tb_senders_00000.dat /path_to_hls4ml/example-prjs/graph/gnn_simple/tb_senders.dat
cp -r firmware/weights /path_to_hls4ml/example-prjs/graph/gnn_simple/.
```

## Set up Vivado HLS
- move to 'hls4ml/example-prjs/graph/gnn_simple` and run:
```
source /path to Xilinx Vivado HLS installation/settings64.sh
```

## Run CSIM (for accuracy test)
- Open `build_prj.tcl` and ensure the csim digit on line six is `1` and the synth digit below it on line seven is `0`
- Save, exit and run:
```
vivado_hls -f build_prj.tcl
```
- To generate Bit Precision vs. AUC plots run:
```
cp -r /path_to_exatrkx_directory/tb_data ./tb_data
python make_roc.py
```

## Run Synthesis (to get RTL and resource estimates)
- Open `build_prj.tcl` and ensure the synth digit on line seven is `1`
- Save, exit, and run:
```
vivado_hls -f build_prj.tcl
```
- The estimates can be found in `myproject_prj/solution1/synth/rpt/myproject_csynth.rpt`