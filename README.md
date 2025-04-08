# Terra Rover - Intelligent Vision System

## Overview

Terra Rover is an intelligent vision system that combines real-time object detection with visual language models (VLM) to provide interactive analysis of live video streams. The system can:

- Process RTSP video streams in real-time
- Detect objects using YOLOv8
- Answer questions about the visual scene using Claude 3 Sonnet
- Accept voice commands via Amazon Lex
- Store and analyze frames in AWS S3

## Key Features

- **Real-time Object Detection**: Powered by YOLOv8 with GPU acceleration
- **Visual Question Answering**: Integrated with Claude 3 for scene understanding
- **Voice Interaction**: Speech recognition via Amazon Lex
- **Cloud Integration**: Automatic frame storage in S3 with lifecycle management
- **Performance Optimized**: Multi-threaded processing with adaptive frame rates

## System Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                        Terra Rover                            │
├───────────────────────────────┬───────────────────────────────┤
│          Vision Stack         │          Voice Stack          │
│                               │                               │
│  RTSP Stream → Object Detection → VLM Processing → S3 Storage │
│                               │                               │
├───────────────────────────────┴───────────────────────────────┤
│                         AWS Integration                       │
│                                                               │
│  S3 (Frame Storage) ↔ Bedrock (VLM) ↔ Lex (Speech Processing) │
└───────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.9+
- NVIDIA GPU with CUDA support (recommended)
- AWS account with Bedrock, Lex, and S3 access
- RTSP video stream source

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/tahirkhan05/terra-rover.git
   cd terra-rover
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment:
   - Copy `.env.example` to `.env`
   - Fill in your AWS credentials and RTSP stream URL
   - Configure other settings as needed

## Usage

```bash
python main.py
```

### Controls

- **'s' key**: Initiate voice query (5-second recording)
- **'q' key**: Quit the application

### Example Interactions

1. Press 's' and ask: "What objects do you see?"
2. The system will:
   - Capture audio and transcribe your question
   - Take a snapshot of the current frame
   - Send to Claude 3 for analysis
   - Display the response in the console

## Configuration

Key configuration options in `.env`:

```ini
# AWS Credentials
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
S3_BUCKET=bucket-name

# RTSP Stream (use IP Webcam app for testing)
RTSP_URL=rtsp://your-ip/h264_pcm.sdp
RTSP_RECONNECT_DELAY=2
RTSP_MAX_RETRIES=10
RTSP_MAX_CONSECUTIVE_FAILURES=30

# High FPS Configuration
FPS=30
FRAME_WIDTH=1280
FRAME_HEIGHT=720

# Model Settings
OBJECT_DETECTION_MODEL=yolov8n.pt
VLM_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0

# Speech Configuration
LEX_BOT_ID=
LEX_BOT_ALIAS_ID=
LEX_LOCALE_ID=en_US

# System Configuration
MAX_WORKERS=4
LOG_LEVEL=DEBUG  
DEBUG=True
LOCAL_SAVE_PATH=data/captured_frames

# Performance Tuning
MAX_QUEUE_SIZE=30
PROCESSING_INTERVAL=0.033  
```

## Performance Considerations

- For best results, use an NVIDIA GPU with CUDA support
- Reduce `FPS` and `FRAME_WIDTH/HEIGHT` for lower-powered systems
- Adjust `MAX_WORKERS` based on your CPU cores

## Troubleshooting

Common issues:

1. **RTSP Connection Failures**:
   - Verify your stream URL is accessible
   - Check firewall settings

2. **AWS Authentication Errors**:
   - Verify IAM permissions for Bedrock, Lex, and S3
   - Check region compatibility

3. **Performance Problems**:
   - Reduce frame resolution in settings
   - Use smaller YOLO model (yolov8n.pt → yolov8s.pt)

## License

Apache License 2.0

## Contributing

Pull requests are welcome! Please open an issue first to discuss major changes.

## Roadmap

- [ ] Add fine tuning according to Dataset
- [ ] Improve FPS

---

*Terra Rover - Seeing and understanding the world in real-time*
