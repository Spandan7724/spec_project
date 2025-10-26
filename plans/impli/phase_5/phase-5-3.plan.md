<!-- f67714c1-a54f-4e8d-9617-16955f212afc a76a2edf-d125-4fa3-9429-4940fc67b340 -->
# Phase 5.3: Next.js + TypeScript Web Frontend Implementation

## Overview

Build a modern, type-safe web frontend using Next.js 14+ with TypeScript, React Server Components, and TailwindCSS. The frontend will consume the FastAPI backend and provide an intuitive chat interface with visualizations.

## Purpose

- Provide modern, responsive web UI for Currency Assistant
- Enable real-time conversation and analysis tracking
- Display interactive charts and visualizations
- Support mobile and desktop browsers
- Type-safe development with TypeScript
- SEO-friendly with Next.js SSR capabilities

## Technology Stack

### Core Framework

- **Next.js 14+**: React framework with App Router
- **TypeScript**: Type safety throughout
- **React 18+**: UI library with Server Components
- **TailwindCSS**: Utility-first CSS framework

### UI Components

- **shadcn/ui**: Accessible component library (Radix UI + Tailwind)
- **Lucide React**: Icon library
- **Framer Motion**: Animations (optional)

### Data & State

- **TanStack Query (React Query)**: API data fetching and caching
- **Zustand**: Lightweight state management (for chat history)
- **Zod**: Runtime type validation

### Visualizations

- **Recharts**: React charts library (for predictions, confidence)
- **Chart.js** + **react-chartjs-2**: Alternative for complex charts
- **Tremor**: Dashboard components (optional)

### API Communication

- **Axios** or **Fetch API**: HTTP requests
- **EventSource**: Server-Sent Events for real-time updates

## Architecture

```
frontend/                              # Next.js project root
├── app/                               # App Router (Next.js 14+)
│   ├── layout.tsx                     # Root layout
│   ├── page.tsx                       # Home page (chat interface)
│   ├── api/                           # API route handlers (optional proxy)
│   │   └── health/
│   │       └── route.ts
│   └── globals.css                    # Global styles
├── components/
│   ├── ui/                            # shadcn/ui components
│   │   ├── button.tsx
│   │   ├── card.tsx
│   │   ├── input.tsx
│   │   ├── badge.tsx
│   │   ├── progress.tsx
│   │   └── ...
│   ├── chat/
│   │   ├── ChatContainer.tsx          # Main chat layout
│   │   ├── ChatMessage.tsx            # Individual message bubble
│   │   ├── ChatInput.tsx              # User input field
│   │   └── ChatHistory.tsx            # Message list
│   ├── analysis/
│   │   ├── AnalysisProgress.tsx       # Progress tracker
│   │   ├── RecommendationCard.tsx     # Main recommendation display
│   │   ├── ConfidenceGauge.tsx        # Confidence visualization
│   │   ├── StagingPlan.tsx            # Tranche timeline
│   │   └── RiskSummary.tsx            # Risk metrics display
│   ├── visualizations/
│   │   ├── PriceChart.tsx             # Historical price chart
│   │   ├── PredictionChart.tsx        # Forecast with confidence bands
│   │   ├── TechnicalIndicators.tsx    # RSI, MACD gauges
│   │   └── CalendarTimeline.tsx       # Economic events
│   └── layout/
│       ├── Header.tsx                 # App header
│       ├── Footer.tsx                 # App footer
│       └── Sidebar.tsx                # Optional sidebar
├── lib/
│   ├── api/
│   │   ├── client.ts                  # Axios/fetch client setup
│   │   ├── conversation.ts            # Conversation API calls
│   │   ├── analysis.ts                # Analysis API calls
│   │   └── visualization.ts           # Viz data API calls
│   ├── hooks/
│   │   ├── useConversation.ts         # Conversation management hook
│   │   ├── useAnalysis.ts             # Analysis execution hook
│   │   ├── useSSE.ts                  # Server-Sent Events hook
│   │   └── useChat.ts                 # Chat state hook
│   ├── store/
│   │   └── chatStore.ts               # Zustand store for chat
│   ├── types/
│   │   ├── api.ts                     # API request/response types
│   │   ├── chat.ts                    # Chat message types
│   │   └── recommendation.ts          # Recommendation types
│   └── utils/
│       ├── formatters.ts              # Currency, percentage formatters
│       ├── validators.ts              # Input validation
│       └── constants.ts               # App constants
├── public/
│   ├── logo.svg
│   └── favicon.ico
├── package.json
├── tsconfig.json
├── tailwind.config.ts
├── next.config.js
└── .env.local                         # Environment variables

tests/
├── components/
│   └── chat.test.tsx
└── lib/
    └── api.test.ts
```

## File Descriptions

### Core App Files

#### `app/layout.tsx`

**Purpose**: Root layout with providers and global styling.

**Components**:

- QueryClientProvider for React Query
- Toaster for notifications
- Global metadata (title, description)
- Font configuration

#### `app/page.tsx`

**Purpose**: Main chat interface page.

**Components**:

- ChatContainer (main chat UI)
- Header with app branding
- Responsive layout (mobile + desktop)

### Chat Components

#### `components/chat/ChatContainer.tsx`

**Purpose**: Main chat interface container.

**Features**:

- Message history display
- Scroll to bottom on new messages
- Auto-focus input field
- Loading states

**State**:

- Messages array (via Zustand)
- Current session ID
- Input value

#### `components/chat/ChatMessage.tsx`

**Purpose**: Individual chat message bubble.

**Props**:

- `message: { role: 'user' | 'assistant', content: string, timestamp: Date }`
- `isLoading?: boolean`

**Features**:

- Different styling for user vs assistant
- Markdown rendering for assistant messages
- Timestamp display
- Loading animation

#### `components/chat/ChatInput.tsx`

**Purpose**: User input field with send button.

**Features**:

- Text input with auto-grow textarea
- Send button with loading state
- Enter to send (Shift+Enter for new line)
- Character limit indicator

**Events**:

- `onSend: (message: string) => void`

#### `components/chat/ChatHistory.tsx`

**Purpose**: Scrollable message list.

**Features**:

- Virtualized list for performance (optional)
- Auto-scroll to bottom
- Date separators
- Typing indicator

### Analysis Components

#### `components/analysis/AnalysisProgress.tsx`

**Purpose**: Display analysis progress during execution.

**Features**:

- Progress bar (0-100%)
- Current step message
- Estimated time remaining
- Cancel button (optional)

**Input**:

- `status: AnalysisStatus` (from API)

#### `components/analysis/RecommendationCard.tsx`

**Purpose**: Main recommendation display card.

**Features**:

- Action badge (CONVERT NOW / STAGED / WAIT)
- Confidence gauge
- Timeline information
- Expandable rationale
- Risk metrics
- Cost estimate

**Input**:

- `recommendation: RecommendationResult`

**Layout**:

- Card header with action
- Body with key info
- Accordion for detailed breakdown

#### `components/analysis/ConfidenceGauge.tsx`

**Purpose**: Visual confidence indicator.

**Features**:

- Circular progress gauge or semi-circle
- Color-coded (green > 70%, yellow > 40%, red < 40%)
- Percentage display
- Component breakdown (optional tooltip)

**Implementation**:

- Recharts RadialBarChart or custom SVG

#### `components/analysis/StagingPlan.tsx`

**Purpose**: Display staged conversion plan.

**Features**:

- Timeline visualization
- Tranche cards with percentages
- Execute day indicators
- Event markers (if aligned with calendar)

**Input**:

- `stagingPlan: { tranches: Array<...>, spacing_days: number }`

#### `components/analysis/RiskSummary.tsx`

**Purpose**: Display risk metrics and warnings.

**Features**:

- Risk level badge
- Key metrics table
- Warning alerts
- Expandable details

**Input**:

- `riskSummary: { risk_level, max_drawdown, var, ... }`

### Visualization Components

#### `components/visualizations/PriceChart.tsx`

**Purpose**: Historical price chart with technical indicators.

**Features**:

- Line/candlestick chart
- Technical indicators overlay (SMA, EMA)
- Zoom and pan
- Responsive sizing

**Library**: Recharts LineChart or Chart.js

**Input**:

- `priceData: Array<{ date, price }>`
- `indicators?: { sma_20, sma_50, ... }`

#### `components/visualizations/PredictionChart.tsx`

**Purpose**: Forecast chart with confidence bands.

**Features**:

- Multi-horizon predictions (1d, 7d, 30d)
- Confidence interval shading
- Current price marker
- Legend

**Library**: Recharts AreaChart

**Input**:

- `predictions: Array<{ horizon, mean, lower, upper }>`

#### `components/visualizations/TechnicalIndicators.tsx`

**Purpose**: Dashboard of technical indicators.

**Features**:

- RSI gauge (0-100)
- MACD bars
- Momentum indicators
- Color-coded signals

**Library**: Recharts or custom progress bars

**Input**:

- `indicators: { rsi, macd, macd_signal, ... }`

#### `components/visualizations/CalendarTimeline.tsx`

**Purpose**: Upcoming economic events timeline.

**Features**:

- Event cards with date/time
- Importance badges (high/medium/low)
- Currency flags/icons
- Countdown timers

**Input**:

- `events: Array<{ date, time, event, importance, currency }>`

### API Layer

#### `lib/api/client.ts`

**Purpose**: Axios/fetch client configuration.

**Features**:

- Base URL from environment
- Default headers
- Request/response interceptors
- Error handling
- Correlation ID injection

#### `lib/api/conversation.ts`

**Purpose**: Conversation API functions.

**Functions**:

- `sendMessage(message: string, sessionId?: string): Promise<ConversationOutput>`
- `resetSession(sessionId: string): Promise<void>`
- `getSession(sessionId: string): Promise<SessionState>`

#### `lib/api/analysis.ts`

**Purpose**: Analysis API functions.

**Functions**:

- `startAnalysis(request: AnalysisRequest): Promise<{ correlation_id: string }>`
- `getStatus(correlationId: string): Promise<AnalysisStatus>`
- `getResult(correlationId: string): Promise<AnalysisResult>`
- `streamProgress(correlationId: string): EventSource`

#### `lib/api/visualization.ts`

**Purpose**: Visualization data API functions.

**Functions**:

- `getConfidenceData(correlationId: string): Promise<ConfidenceData>`
- `getTechnicalIndicators(currencyPair: string): Promise<TechnicalData>`
- `getCalendarEvents(currency: string): Promise<CalendarEvents>`
- `getPriceHistory(currencyPair: string, days: number): Promise<PriceHistory>`

### Hooks

#### `lib/hooks/useConversation.ts`

**Purpose**: React Query hook for conversation management.

**Returns**:

- `sendMessage: (message: string) => void`
- `messages: Message[]`
- `sessionId: string`
- `isLoading: boolean`
- `error: Error | null`

**Implementation**:

- Uses TanStack Query mutations
- Manages session state
- Handles optimistic updates

#### `lib/hooks/useAnalysis.ts`

**Purpose**: Hook for analysis execution and polling.

**Returns**:

- `startAnalysis: (params: AnalysisParams) => void`
- `status: AnalysisStatus`
- `result: AnalysisResult | null`
- `progress: number`
- `isAnalyzing: boolean`

**Implementation**:

- Starts analysis
- Polls status endpoint
- Fetches final result
- Uses React Query

#### `lib/hooks/useSSE.ts`

**Purpose**: Server-Sent Events hook for real-time updates.

**Returns**:

- `connect: (url: string) => void`
- `disconnect: () => void`
- `data: any[]`
- `isConnected: boolean`

**Implementation**:

- EventSource API wrapper
- Auto-reconnect logic
- Cleanup on unmount

#### `lib/hooks/useChat.ts`

**Purpose**: Chat state management hook.

**Returns**:

- `messages: Message[]`
- `addMessage: (message: Message) => void`
- `clearMessages: () => void`
- `sessionId: string`

**Implementation**:

- Wraps Zustand store
- Persists to localStorage (optional)

### State Management

#### `lib/store/chatStore.ts`

**Purpose**: Zustand store for chat state.

**State**:

- `messages: Message[]`
- `sessionId: string | null`
- `currentInput: string`

**Actions**:

- `addMessage(message: Message)`
- `setSessionId(id: string)`
- `setInput(value: string)`
- `clearChat()`

### Type Definitions

#### `lib/types/api.ts`

**Purpose**: API request/response types matching FastAPI backend.

**Types**:

- `ConversationInput`
- `ConversationOutput`
- `AnalysisRequest`
- `AnalysisStatus`
- `AnalysisResult`
- `HealthResponse`

**Implementation**:

- Mirror Pydantic models from backend
- Use Zod for runtime validation

#### `lib/types/chat.ts`

**Purpose**: Chat-specific types.

**Types**:

- `Message: { id, role, content, timestamp, metadata? }`
- `ChatState: { messages, sessionId, isTyping }`

#### `lib/types/recommendation.ts`

**Purpose**: Recommendation data types.

**Types**:

- `Recommendation: { action, confidence, timeline, rationale, ... }`
- `StagingPlan: { tranches, spacing_days }`
- `RiskSummary: { risk_level, metrics }`

### Utilities

#### `lib/utils/formatters.ts`

**Purpose**: Formatting functions.

**Functions**:

- `formatCurrency(amount: number, currency: string): string`
- `formatPercentage(value: number, decimals?: number): string`
- `formatDate(date: Date): string`
- `formatConfidence(confidence: number): string`

#### `lib/utils/validators.ts`

**Purpose**: Input validation.

**Functions**:

- `validateCurrencyPair(pair: string): boolean`
- `validateAmount(amount: string): boolean`
- `validateSessionId(id: string): boolean`

## Key Features

### 1. Real-Time Chat Interface

- Smooth message animations
- Typing indicators
- Markdown rendering for assistant messages
- Auto-scroll to latest message

### 2. Progressive Analysis Display

- Show progress as agents execute
- Real-time updates via SSE or polling
- Smooth transitions to recommendation

### 3. Rich Visualizations

- Interactive charts with tooltips
- Responsive sizing
- Color-coded confidence levels
- Economic calendar timeline

### 4. Responsive Design

- Mobile-first approach
- Desktop optimizations
- Touch-friendly interactions
- Adaptive layouts

### 5. Type Safety

- Full TypeScript coverage
- Zod validation for API responses
- Compile-time error checking

## Implementation Steps

1. **Initialize Next.js project**
   ```bash
   cd frontend  # or create new directory
   npx create-next-app@latest . --typescript --tailwind --app --src-dir=false
   ```

2. **Install dependencies**
   ```bash
   npm install @tanstack/react-query zustand zod axios
   npm install recharts lucide-react
   npm install -D @types/node
   ```

3. **Setup shadcn/ui**
   ```bash
   npx shadcn-ui@latest init
   npx shadcn-ui@latest add button card input badge progress
   ```

4. **Configure environment variables** (`.env.local`)
   ```
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

5. **Implement API client** (`lib/api/client.ts`)

6. **Create type definitions** (`lib/types/*.ts`)

7. **Implement API functions** (`lib/api/*.ts`)

8. **Create custom hooks** (`lib/hooks/*.ts`)

9. **Implement Zustand store** (`lib/store/chatStore.ts`)

10. **Build UI components** (progressively: chat → analysis → visualizations)

11. **Implement pages** (`app/page.tsx`)

12. **Add styling and animations**

13. **Write tests** (Vitest or Jest + React Testing Library)

14. **Test with backend** (integration testing)

## Dependencies

```json
{
  "dependencies": {
    "next": "^14.0.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "typescript": "^5.3.0",
    "@tanstack/react-query": "^5.0.0",
    "zustand": "^4.4.0",
    "zod": "^3.22.0",
    "axios": "^1.6.0",
    "recharts": "^2.10.0",
    "lucide-react": "^0.300.0",
    "tailwindcss": "^3.4.0",
    "clsx": "^2.0.0",
    "tailwind-merge": "^2.0.0"
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "eslint": "^8.0.0",
    "eslint-config-next": "^14.0.0",
    "vitest": "^1.0.0",
    "@testing-library/react": "^14.0.0"
  }
}
```

## Running the Frontend

### Development

```bash
cd frontend
npm install
npm run dev

# Access at http://localhost:3000
```

### Production Build

```bash
npm run build
npm run start
```

### Docker (Optional)

```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build
CMD ["npm", "start"]
```

## Environment Variables

```bash
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

## Testing Strategy

- **Unit tests**: Test utility functions and hooks
- **Component tests**: Test UI components in isolation
- **Integration tests**: Test complete user flows
- **E2E tests**: Playwright or Cypress (optional)

## Success Criteria

- Chat interface works smoothly with multi-turn conversation
- Parameters are extracted and confirmed visually
- Analysis progress displays in real-time
- Recommendations render with all visualizations
- Charts are interactive and responsive
- Mobile experience is smooth
- TypeScript has no errors
- All components are accessible (ARIA)
- Fast page loads (< 2s initial load)

## Integration Points

- FastAPI backend at `http://localhost:8000`
- `/api/conversation/message` for chat
- `/api/analysis/start` for initiating analysis
- `/api/analysis/status` for progress tracking
- `/api/analysis/result` for final recommendation
- `/api/viz/*` for visualization data
- `/health` for health checks

## Styling Guide

- Use TailwindCSS utility classes
- Follow shadcn/ui design patterns
- Color scheme:
  - Primary: Blue (actions, links)
  - Success: Green (high confidence, positive)
  - Warning: Yellow (medium confidence, caution)
  - Error: Red (low confidence, errors)
  - Neutral: Gray (text, borders)

## Accessibility

- Semantic HTML tags
- ARIA labels for interactive elements
- Keyboard navigation support
- Screen reader compatibility
- Focus indicators
- Color contrast ratios (WCAG AA)

## Performance Optimizations

- React Server Components where possible
- Code splitting for charts
- Image optimization with Next.js Image
- Lazy loading for visualizations
- Request deduplication with React Query
- Memoization for expensive computations

## Notes

- Next.js App Router (not Pages Router)
- Use Server Components by default, Client Components only when needed
- API calls should use absolute URLs (from env variable)
- Consider adding authentication later (NextAuth.js)
- Consider adding i18n for multiple languages
- Use React Suspense for loading states
- Consider adding error boundaries for graceful error handling

### To-dos

- [ ] Initialize Next.js 14+ project with TypeScript and TailwindCSS
- [ ] Install and configure shadcn/ui component library
- [ ] Create API client layer with axios/fetch in lib/api/
- [ ] Implement TypeScript types for API requests/responses in lib/types/
- [ ] Create React Query hooks for data fetching (useConversation, useAnalysis)
- [ ] Implement Zustand store for chat state management
- [ ] Build chat UI components (ChatContainer, ChatMessage, ChatInput)
- [ ] Build analysis components (RecommendationCard, ConfidenceGauge, StagingPlan, RiskSummary)
- [ ] Implement visualization components with Recharts (PriceChart, PredictionChart)
- [ ] Write component tests with React Testing Library
- [ ] Test full user flow end-to-end with FastAPI backend