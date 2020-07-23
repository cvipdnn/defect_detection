# CNN Search based on MobileNetV2 

## Usage

python searchCNN_MobileNet.py

Apply feature output from different layers in MobileNetV2 and figure out which option can reach accuracy requirement with less computation.  


featurelayers=['out_relu', 'block_15_depthwise_relu', 'block_13_depthwise_relu', 'block_13_expand_relu', 'block_10_depthwise_relu', 'block_8_depthwise_relu', 'block_6_depthwise_relu', 'block_5_depthwise_relu', 'block_3_depthwise_relu','block_1_depthwise_relu']

# Result
![](performance.PNG)
