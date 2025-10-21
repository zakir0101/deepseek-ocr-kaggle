# Vast.ai Deployment Guide for DeepSeek OCR

## Overview
Vast.ai is a GPU rental platform that provides affordable access to high-performance GPUs for AI/ML workloads. This guide covers how to deploy the DeepSeek OCR model on vast.ai instances.

## Key Concepts

### Instance Types and Pricing
- **Target GPUs**: RTX 3080, RTX 3090, RTX 4090, A100 (compute capability ≥ 8.0)
- **Price Range**: $0.10 - $0.20 per hour
- **Minimum Requirements**:
  - 16GB GPU RAM
  - 16GB System RAM
  - 50GB Storage
  - Good host reviews and fast internet

### Instance Selection Criteria
1. **GPU Compute Capability**: Must be ≥ 8.0 for bfloat16 support
2. **Host Reliability**: Look for hosts with high uptime and good reviews
3. **Network Speed**: Prefer hosts with fast internet connections
4. **Location**: Choose geographically close hosts for lower latency

## Deployment Architecture

### File System Structure
```
/opt/workspace-internal/     # Main workspace (files sync automatically)
├── deepseek-ocr-model/      # Model files
├── uploads/                 # Uploaded images
├── outputs/                 # OCR results
└── scripts/                 # Deployment scripts
```

### Port Configuration
- **Web Server**: Port 5000 (internal) → Mapped to external port
- **Health Check**: Port 8080 (internal)
- **Ngrok Tunnel**: Automatic public URL generation

## Code Optimizations for Vast.ai

### Removed Kaggle-Specific Code
- Removed `/kaggle/working` path references
- Simplified model persistence logic
- Removed Kaggle environment checks

### Vast.ai Specific Features
- Added `PORTAL_CONFIG` for web interface access
- Environment variable-based configuration
- SSH key management support
- Automatic file synchronization

## Deployment Scripts

### vast_setup.py
```python
# Simplified setup script for vast.ai
# - Downloads model to /opt/workspace-internal/
# - Installs dependencies
# - Configures environment
```

### vast_server.py
```python
# Optimized server for vast.ai
# - Uses /opt/workspace-internal/ paths
# - Auto-detects GPU capabilities
# - Configures ngrok for public access
```

## Environment Configuration

### Required Environment Variables
```bash
MODEL_PATH=/opt/workspace-internal/deepseek-ocr-model
UPLOAD_FOLDER=/opt/workspace-internal/uploads
OUTPUT_FOLDER=/opt/workspace-internal/outputs
PORTAL_CONFIG="localhost:5000:15000:/:DeepSeek OCR Server"
```

### PORTAL_CONFIG Format
```
localhost:INTERNAL_PORT:EXTERNAL_PORT:/PATH:APP_NAME
```
Example: `localhost:5000:15000:/:DeepSeek OCR Server`

## Provisioning Script (On-Start)

```bash
#!/bin/bash
cd /opt/workspace-internal/

# Activate virtual environment
. /venv/main/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download model if not exists
python vast_setup.py

# Start server
python vast_server.py
```

## Vast.ai CLI Usage

### Installation and Setup
```bash
# Install vast.ai CLI
pip install vastai

# Set API key (get from vast.ai website)
vastai set api-key YOUR_API_KEY_HERE
```

### Searching for GPU Instances
```bash
# Search for GPUs with compute capability > 8.0 (bfloat16 support)
vastai search offers 'compute_cap > 800 gpu_ram > 16 num_gpus = 1'

# Search for specific GPU models
vastai search offers 'gpu_name in (RTX_3080, RTX_3090, RTX_4090, A100)'

# Search with price filter (≤ $0.20/hour)
vastai search offers 'dph <= 0.20 compute_cap > 800'

# Search with reliability filter
vastai search offers 'reliability > 0.99 compute_cap > 800'

# Combined search for optimal instances
vastai search offers 'compute_cap > 800 gpu_ram > 16 num_gpus = 1 dph <= 0.20 reliability > 0.95'
```

### Instance Creation
```bash
# Basic instance creation with PyTorch image
vastai create instance <offer_id> --image pytorch/pytorch:latest --disk 50 --ssh --direct

# With custom environment variables and port mapping
vastai create instance <offer_id> \
  --image pytorch/pytorch:latest \
  --env '-p 5000:5000 -p 8080:8080 -e PORTAL_CONFIG="localhost:5000:15000:/:DeepSeek OCR Server"' \
  --disk 50 \
  --ssh \
  --direct

# With custom Docker options
vastai create instance <offer_id> \
  --image pytorch/pytorch:latest \
  --env '-p 5000:5000 -p 8080:8080 -e MODEL_PATH=/opt/workspace-internal/deepseek-ocr-model' \
  --disk 50 \
  --ssh \
  --direct
```

### Instance Management
```bash
# Show current instances
vastai show instances

# Start/stop instances
vastai start instance <instance_id>
vastai stop instance <instance_id>

# Destroy instance (irreversible)
vastai destroy instance <instance_id>

# Copy files to/from instances
vastai copy ~/local/files <instance_id>:/opt/workspace-internal/
vastai copy <instance_id>:/opt/workspace-internal/outputs ~/local/downloads/

# Get instance logs
vastai logs <instance_id>
```

### SSH Access
```bash
# SSH into instance (port and IP from instance details)
ssh -p <port> root@<ip_address>

# With port forwarding for web access
ssh -p <port> root@<ip_address> -L 8080:localhost:8080 -L 5000:localhost:5000
```
```

## Cost Optimization

### Instance Selection Tips
1. **Bid Pricing**: Start with $0.15/hour and adjust based on availability
2. **Spot Instances**: Use spot pricing for 50-70% cost savings
3. **Auto-stop**: Configure auto-stop after inactivity
4. **Storage**: Use instance storage for temporary files

### Monitoring
- Monitor GPU utilization with `nvidia-smi`
- Track costs in vast.ai dashboard
- Set up alerts for high usage

## Troubleshooting

### Common Issues
1. **GPU Compatibility**: Ensure compute capability ≥ 8.0
2. **Port Conflicts**: Check PORTAL_CONFIG mappings
3. **File Sync**: Use `/opt/workspace-internal/` for automatic sync
4. **Dependencies**: Verify CUDA and PyTorch versions

### Performance Optimization
- Use bfloat16 on compatible GPUs (compute capability ≥ 8.0)
- Enable GPU memory optimization
- Configure appropriate batch sizes
- Monitor memory usage and adjust accordingly

## Security Considerations

1. **SSH Keys**: Use strong SSH keys for instance access
2. **Ngrok Tokens**: Secure ngrok authentication tokens
3. **Port Security**: Limit exposed ports to necessary services
4. **Data Privacy**: Ensure sensitive data is properly handled

## Cost Estimation

- **GPU Instance**: $0.15 - $0.20 per hour
- **Storage**: Included in instance cost
- **Network**: Included in instance cost
- **Total**: ~$3.60 - $4.80 per day (24h runtime)

This deployment approach provides a cost-effective, scalable solution for running DeepSeek OCR with proper GPU support and public accessibility.