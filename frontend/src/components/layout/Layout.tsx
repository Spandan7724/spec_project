import { Link, Outlet, useLocation } from 'react-router-dom';
import { Home, MessageSquare, BarChart3, Cpu, Moon, Sun, History } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';

export default function Layout() {
  const location = useLocation();
  const { theme, toggleTheme } = useTheme();

  const isActive = (path: string) => location.pathname === path;
  const isChatPage = location.pathname === '/chat';

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Only show nav if not on chat page */}
      {!isChatPage && (
        <nav className="border-b">
          <div className="container mx-auto px-4 py-4">
            <div className="flex items-center justify-between">
              <Link to="/" className="text-xl font-bold">
                Forex
              </Link>

              <div className="flex items-center gap-6">
                <div className="flex gap-6">
                  <Link
                    to="/"
                    className={`flex items-center gap-2 transition-colors ${
                      isActive('/') ? 'text-primary' : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    <Home size={20} />
                    <span>Home</span>
                  </Link>

                  <Link
                    to="/chat"
                    className={`flex items-center gap-2 transition-colors ${
                      isActive('/chat') ? 'text-primary' : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    <MessageSquare size={20} />
                    <span>Chat</span>
                  </Link>

                  <Link
                    to="/analysis"
                    className={`flex items-center gap-2 transition-colors ${
                      isActive('/analysis') ? 'text-primary' : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    <BarChart3 size={20} />
                    <span>Analysis</span>
                  </Link>

                  <Link
                    to="/models"
                    className={`flex items-center gap-2 transition-colors ${
                      isActive('/models') ? 'text-primary' : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    <Cpu size={20} />
                    <span>Models</span>
                  </Link>

                  <Link
                    to="/history"
                    className={`flex items-center gap-2 transition-colors ${
                      isActive('/history') ? 'text-primary' : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    <History size={20} />
                    <span>History</span>
                  </Link>
                </div>

                <button
                  onClick={toggleTheme}
                  className="p-2 rounded-lg hover:bg-accent transition-colors"
                  aria-label="Toggle theme"
                >
                  {theme === 'light' ? <Moon size={20} /> : <Sun size={20} />}
                </button>
              </div>
            </div>
          </div>
        </nav>
      )}

      {/* Conditional main wrapper - no padding for chat */}
      {isChatPage ? (
        <Outlet />
      ) : (
        <main className="container mx-auto px-4 py-8">
          <Outlet />
        </main>
      )}
    </div>
  );
}
