# Smart Summarizer

AI-powered application for summarizing YouTube videos and PDF documents using Google Gemini AI.

## 🌟 Features

- **YouTube Video Summarization**: Extract and summarize transcripts from YouTube videos
- **PDF Document Summarization**: Upload and summarize PDF documents
- **Multiple Summary Formats**:
  - Short Summary (3-5 paragraphs)
  - Bullet Points (customizable count)
  - Detailed Summary (comprehensive with explanations)
- **Advanced AI**: Powered by Google Gemini 2.0 Flash
- **Modern UI**: Clean, responsive interface
- **Free Deployment**: Optimized for Render.com free tier

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Google Gemini API Key ([Get it here](https://aistudio.google.com/app/apikey))

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Smart-Summarizer.git
   cd Smart-Summarizer
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Copy `.env.example` to `.env`
   - Add your Gemini API key:
     ```
     GEMINI_API_KEY=your_actual_api_key_here
     ```

5. **Run the application**
   ```bash
   uvicorn main:app --reload
   ```

6. **Open your browser**
   - Navigate to: `http://localhost:8000`

## 📦 Deployment

### Deploy to Render.com (Free)

**Full deployment instructions are in [RENDER_DEPLOYMENT_GUIDE.md](RENDER_DEPLOYMENT_GUIDE.md)**

Quick summary:
1. Push code to GitHub
2. Create new Web Service on Render
3. Connect your repository
4. Configure environment variables
5. Deploy!

Your app will be live at: `https://your-app-name.onrender.com`

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python)
- **AI Model**: Google Gemini 2.0 Flash
- **YouTube**: yt-dlp, youtube-transcript-api
- **PDF Processing**: pdfplumber
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Render.com

## 📁 Project Structure

```
Smart-Summarizer/
├── main.py                    # FastAPI backend application
├── index.html                 # Frontend UI
├── script.js                  # Frontend JavaScript
├── styles.css                 # Frontend styles
├── requirements.txt           # Python dependencies
├── Procfile                   # Render start command
├── render.yaml                # Render configuration
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose configuration
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
└── RENDER_DEPLOYMENT_GUIDE.md # Deployment instructions
```

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key |
| `YTDLP_COOKIES` | No | Path to cookies.txt for age-restricted videos |
| `HOST` | No | Server host (default: 0.0.0.0) |
| `PORT` | No | Server port (default: 8000) |
| `DEBUG` | No | Debug mode (default: False) |

## 📝 API Endpoints

### GET `/`
- Serves the main HTML interface

### GET `/health`
- Health check endpoint
- Returns application status and configuration

### POST `/summarize/youtube`
- Summarizes a YouTube video
- **Form Data:**
  - `video_url` (required): YouTube video URL
  - `summary_type` (optional): "short", "bullet", or "detailed"
  - `bullet_count` (optional): Number of bullet points (if type is "bullet")
  - `target_lang` (optional): Target language (default: "en")

### POST `/summarize/pdf`
- Summarizes a PDF document
- **Form Data:**
  - `file` (required): PDF file upload
  - `summary_type` (optional): "short", "bullet", or "detailed"
  - `bullet_count` (optional): Number of bullet points (if type is "bullet")
  - `target_lang` (optional): Target language (default: "en")

## 🎯 Usage Examples

### YouTube Video Summarization

1. Go to the application URL
2. Click "YouTube Video" tab
3. Paste a YouTube URL
4. Select summary type
5. Click "Summarize"

### PDF Document Summarization

1. Go to the application URL
2. Click "PDF Document" tab
3. Upload a PDF file
4. Select summary type
5. Click "Summarize"

## 🚨 Troubleshooting

### Common Issues

**YouTube videos not summarizing:**
- Check if the video has captions/transcripts
- Wait a few minutes if YouTube is rate-limiting
- Try a different video

**PDF upload fails:**
- Ensure PDF is not corrupted
- Try a smaller file (under 10 MB recommended)
- Check PDF is text-based (not scanned images)

**API errors:**
- Verify `GEMINI_API_KEY` is set correctly
- Check API quota in Google AI Studio
- Review application logs

### Render Free Tier Notes

- **Cold Start**: First request after 15 minutes takes ~30-60 seconds
- **RAM Limit**: 512 MB (application is optimized for this)
- **Sleep Mode**: App sleeps after 15 minutes of inactivity

## 🔐 Security

- Never commit `.env` file to version control
- Use environment variables for sensitive data
- Keep API keys secure
- Rotate API keys regularly

## 📊 Performance

### Optimizations for Free Tier:
- Single worker process
- Request concurrency limits
- Optimized timeout settings
- Efficient memory usage
- Smart chunking for large documents

### Processing Times:
- Short videos (< 10 min): 10-20 seconds
- Long videos (> 30 min): 30-60 seconds
- Small PDFs (< 50 pages): 15-30 seconds
- Large PDFs (> 100 pages): 45-90 seconds

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👥 Credits

Developed at **Hirasugar Institute of Technology**

### Technologies Used:
- [FastAPI](https://fastapi.tiangolo.com/)
- [Google Gemini AI](https://ai.google.dev/)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [pdfplumber](https://github.com/jsvine/pdfplumber)
- [Render.com](https://render.com/)

## 📞 Support

For issues or questions:
- Check the troubleshooting section
- Review the deployment guide
- Check application logs
- Open a GitHub issue

## 🔄 Version History

### v1.3.2 (Current)
- Optimized for Render free tier deployment
- Improved error handling
- Enhanced YouTube extraction
- Better memory management
- Updated dependencies

### v1.3.0
- Multiple extraction strategies
- Bot detection bypass
- Improved formatting
- Enhanced error messages

## 🎓 Educational Use

This project is ideal for:
- Learning FastAPI development
- Understanding AI integration
- Practicing deployment workflows
- Building practical web applications

---

**Made with ❤️ at Hirasugar Institute of Technology**

**Version**: 1.3.2  
**Last Updated**: November 2025
