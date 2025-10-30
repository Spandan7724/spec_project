import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './contexts/ThemeContext';
import { SessionProvider } from './contexts/SessionContext';
import Layout from './components/layout/Layout';
import Home from './pages/Home';
import Chat from './pages/Chat';
import Analysis from './pages/Analysis';
import Results from './pages/Results';
import Models from './pages/Models';
import History from './pages/History';

function App() {
  return (
    <ThemeProvider>
      <SessionProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Layout />}>
              <Route index element={<Home />} />
              <Route path="chat" element={<Chat />} />
              <Route path="analysis" element={<Analysis />} />
              <Route path="results/:correlationId" element={<Results />} />
              <Route path="models" element={<Models />} />
              <Route path="history" element={<History />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </SessionProvider>
    </ThemeProvider>
  );
}

export default App;
