# Currency Assistant Frontend

React + Vite + TypeScript frontend for the Currency Assistant application.

## Tech Stack

- **React 18** - UI library
- **Vite** - Build tool & dev server
- **TypeScript** - Type safety
- **Tailwind CSS** - Utility-first styling
- **Shadcn/ui** - UI component library
- **React Router** - Client-side routing
- **Axios** - HTTP client
- **Recharts** - Charts and visualizations
- **Lucide React** - Icons

## Features

✅ **Chat Interface** - Conversational UI for guided parameter collection
✅ **Quick Analysis Form** - Form-based analysis submission
✅ **Results Dashboard** - Real-time progress with SSE streaming
✅ **Comprehensive Results** - Recommendation, confidence, risk, costs, timeline
✅ **Model Training UI** - Train and manage LightGBM/LSTM models
✅ **Responsive Design** - Mobile-friendly layouts

## Getting Started

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

Runs on http://localhost:5173

## Environment Variables

Create a `.env` file:

```env
VITE_API_BASE_URL=http://localhost:8000
```

## Available Routes

- `/` - Home page with feature cards
- `/chat` - Chat interface
- `/analysis` - Quick analysis form
- `/results/:correlationId` - Analysis results dashboard
- `/models` - Model training & management
