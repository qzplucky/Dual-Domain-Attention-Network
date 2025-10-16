from graphviz import Digraph

# 初始化图形
dot = Digraph(comment='U-Net Structure', node_attr={'shape': 'box'})

# 编码器层
enc_layers = ['Conv1 (64)', 'Max Pool', 'Conv2 (128)', 'Max Pool', 
              'Conv3 (256)', 'Max Pool', 'Conv4 (512)', 'Max Pool', 'Conv5 (1024)']
for i, layer in enumerate(enc_layers):
    dot.node(f'enc{i}', layer)

# 解码器层
dec_layers = ['UpConv4 (512)', 'Conv4 (512)', 'UpConv3 (256)', 'Conv3 (256)', 
              'UpConv2 (128)', 'Conv2 (128)', 'UpConv1 (64)', 'Conv1 (64)', 'Output (1)']
for i, layer in enumerate(dec_layers):
    dot.node(f'dec{i}', layer)

# 连接编码器
for i in range(len(enc_layers)-1):
    dot.edge(f'enc{i}', f'enc{i+1}')

# 连接解码器
for i in range(len(dec_layers)-1):
    dot.edge(f'dec{i}', f'dec{i+1}')

# 跳跃连接（示例：enc1 → dec6, enc3 → dec4, enc5 → dec2）
jump_edges = [('enc1', 'dec6'), ('enc3', 'dec4'), ('enc5', 'dec2')]
for src, dst in jump_edges:
    dot.edge(src, dst, style='dashed')  # 虚线表示跳跃连接

# 保存为 PDF/SVG
dot.render('unet_structure', format='pdf', cleanup=True)