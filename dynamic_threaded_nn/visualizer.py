import torch
from typing import List, Dict, Optional
import os

def render_ascii_threads(graph_info: Dict[str, torch.Tensor], sample_idx: int = 0) -> str:
    """
    Renders an ASCII connectivity chart of dynamic threads for a chosen sample.
    """
    weights = graph_info["thread_weights"][sample_idx].detach().cpu().numpy()
    mask = graph_info["active_mask"][sample_idx].detach().cpu().numpy()
    layer_names = graph_info["layer_names"]
    num_nodes = len(layer_names)
    
    lines = []
    lines.append("=" * 60)
    lines.append(f"DYNAMIC THREAD CONNECTIVITY GRAPH (Sample #{sample_idx})")
    lines.append("=" * 60)
    
    for i in range(num_nodes):
        active_destinations = []
        for j in range(i + 1, num_nodes):
            w = weights[i, j]
            m = mask[i, j]
            if m > 0 or w > 0.05:
                active_destinations.append(f"-> {layer_names[j]} [weight={w:.3f}]")
        
        lines.append(f"[{layer_names[i]}]")
        if active_destinations:
            for dest in active_destinations:
                lines.append(f"    |--> (Dynamic Thread) {dest}")
        else:
            lines.append("    |--> (No active downstream threads)")
    
    lines.append("=" * 60)
    return "\n".join(lines)


def export_interactive_html(
    graph_info: Dict[str, torch.Tensor], 
    sample_indices: List[int] = [0, 1, 2],
    output_path: str = "dynamic_threads_visualization.html"
) -> str:
    """
    Generates an interactive, beautiful HTML5 SVG visualizer displaying dynamic threads
    pulsing between neural network layers with slider/sample controls.
    """
    layer_names = graph_info["layer_names"]
    num_nodes = len(layer_names)
    
    # Extract weights for the selected samples
    samples_data = []
    for s_idx in sample_indices:
        w_mat = graph_info["thread_weights"][s_idx].detach().cpu().tolist()
        m_mat = graph_info["active_mask"][s_idx].detach().cpu().tolist()
        samples_data.append({
            "sample_index": s_idx,
            "weights": w_mat,
            "mask": m_mat
        })
        
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Dynamic Threaded Neural Network Visualization</title>
<style>
  body {{
    margin: 0;
    padding: 24px;
    background: #0f172a;
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  h1 {{
    color: #38bdf8;
    margin-bottom: 8px;
    font-size: 24px;
  }}
  .subtitle {{
    color: #94a3b8;
    margin-bottom: 20px;
    font-size: 14px;
  }}
  .controls {{
    margin-bottom: 20px;
    display: flex;
    gap: 12px;
    align-items: center;
  }}
  select, button {{
    background: #1e293b;
    color: #f8fafc;
    border: 1px solid #475569;
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
  }}
  svg {{
    background: #111827;
    border-radius: 12px;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
    width: 100%;
    max-width: 1000px;
    height: 480px;
  }}
  .layer-box {{
    fill: #1e293b;
    stroke: #38bdf8;
    stroke-width: 2;
    rx: 8;
  }}
  .layer-title {{
    fill: #f8fafc;
    font-size: 13px;
    font-weight: 600;
    text-anchor: middle;
  }}
  .thread-path {{
    fill: none;
    transition: all 0.4s ease;
  }}
  .stats {{
    margin-top: 16px;
    background: #1e293b;
    padding: 16px;
    border-radius: 8px;
    max-width: 1000px;
    font-size: 14px;
    border-left: 4px solid #38bdf8;
  }}
</style>
</head>
<body>
  <h1>Dynamic Threaded Neural Network</h1>
  <div class="subtitle">Dynamic connection threads routed adaptively per input sample</div>
  
  <div class="controls">
    <label for="sampleSelect">Select Input Sample:</label>
    <select id="sampleSelect" onchange="renderSample(this.value)">
      {"".join([f'<option value="{idx}">Sample #{idx}</option>' for idx in range(len(samples_data))])}
    </select>
  </div>

  <svg id="networkSvg" viewBox="0 0 1000 480"></svg>
  
  <div id="statsBox" class="stats"></div>

  <script>
    const layerNames = {layer_names};
    const numLayers = {num_nodes};
    const samples = {samples_data};
    const svg = document.getElementById('networkSvg');
    const statsBox = document.getElementById('statsBox');

    function renderSample(sampleIdx) {{
      const data = samples[sampleIdx];
      const weights = data.weights;
      const mask = data.mask;
      
      svg.innerHTML = '';
      
      const width = 1000;
      const height = 480;
      const layerSpacing = width / (numLayers + 1);
      
      // Node positions
      const layerPositions = [];
      for (let i = 0; i < numLayers; i++) {{
        const x = (i + 1) * layerSpacing;
        const y = height / 2;
        layerPositions.push({{ x, y }});
      }}
      
      // Draw dynamic threads (bezier curves)
      let activeThreadCount = 0;
      let threadSummaries = [];

      for (let i = 0; i < numLayers; i++) {{
        for (let j = i + 1; j < numLayers; j++) {{
          const weight = weights[i][j];
          const isMasked = mask[i][j] > 0;
          
          if (weight > 0.01) {{
            activeThreadCount++;
            const p1 = layerPositions[i];
            const p2 = layerPositions[j];
            const dx = (p2.x - p1.x) / 2;
            const dy = (j - i > 1) ? ((j % 2 === 0 ? 1 : -1) * (j - i) * 35) : 0;
            const cy = height / 2 + dy;
            
            const pathData = `M ${{p1.x + 40}} ${{p1.y}} Q ${{p1.x + dx}} ${{cy}} ${{p2.x - 40}} ${{p2.y}}`;
            
            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.setAttribute('d', pathData);
            path.setAttribute('class', 'thread-path');
            
            // Dynamic color and glow based on weight
            const alpha = Math.min(1.0, Math.max(0.2, weight * 1.5));
            const strokeWidth = Math.max(1.5, weight * 6);
            const color = (j - i > 1) ? `rgba(244, 63, 94, ${{alpha}})` : `rgba(56, 189, 248, ${{alpha}})`;
            
            path.setAttribute('stroke', color);
            path.setAttribute('stroke-width', strokeWidth);
            
            if (isMasked) {{
              path.setAttribute('stroke-dasharray', 'none');
            }} else {{
              path.setAttribute('stroke-dasharray', '4 4');
            }}
            
            svg.appendChild(path);
            threadSummaries.push(`${{layerNames[i]}} &rarr; ${{layerNames[j]}} (wt: ${{weight.toFixed(3)}})`);
          }}
        }}
      }}
      
      // Draw layer nodes
      for (let i = 0; i < numLayers; i++) {{
        const pos = layerPositions[i];
        
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', pos.x - 45);
        rect.setAttribute('y', pos.y - 45);
        rect.setAttribute('width', 90);
        rect.setAttribute('height', 90);
        rect.setAttribute('class', 'layer-box');
        
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', pos.x);
        text.setAttribute('y', pos.y + 5);
        text.setAttribute('class', 'layer-title');
        text.textContent = layerNames[i];
        
        g.appendChild(rect);
        g.appendChild(text);
        svg.appendChild(g);
      }}

      statsBox.innerHTML = `<strong>Active Dynamic Threads:</strong> ${{activeThreadCount}}<br>` +
                           `<strong>Active Connections:</strong> ${{threadSummaries.join(' | ')}}`;
    }}

    // Initial render
    renderSample(0);
  </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return output_path
