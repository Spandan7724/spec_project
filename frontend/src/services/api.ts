import axios from 'axios';

// Prefer relative API in dev so Vite can proxy to backend, avoiding CORS/IPv6 issues.
// Set VITE_API_BASE_URL only for production builds or special setups.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

export default api;
