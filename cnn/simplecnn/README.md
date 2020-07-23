# CNN Search based on MobileNetV2 

Apply feature output from different layers in MobileNetV2 and figure out which one can reach accuracy requirement with less computation.  

featurelayers=['out_relu', 'block_15_depthwise_relu', 'block_13_depthwise_relu', 'block_13_expand_relu', 'block_10_depthwise_relu', 'block_8_depthwise_relu', 'block_6_depthwise_relu', 'block_5_depthwise_relu', 'block_3_depthwise_relu','block_1_depthwise_relu']

![](performance.png)
