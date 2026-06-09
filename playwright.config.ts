import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "tests/gpu",
  webServer: { command: "npm run dev", url: "http://localhost:5173", reuseExistingServer: true },
  use: {
    baseURL: "http://localhost:5173",
    launchOptions: {
      args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=metal"],
    },
  },
});
