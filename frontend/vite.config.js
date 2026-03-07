import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(() => {
  const backendOrigin = process.env.VITE_BACKEND_ORIGIN || "http://127.0.0.1:8000";
  const proxyWithCsrfOrigin = {
    target: backendOrigin,
    changeOrigin: true,
    headers: {
      Origin: backendOrigin
    }
  };

  return {
    plugins: [react()],
    base: "/static/",
    server: {
      host: "127.0.0.1",
      proxy: {
        "/api": proxyWithCsrfOrigin,
        "/media": proxyWithCsrfOrigin
      }
    },
    build: {
      outDir: "dist",
      emptyOutDir: true
    }
  };
});
