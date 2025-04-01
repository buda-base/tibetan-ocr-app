document.addEventListener("DOMContentLoaded", function () {
  // Fetch latest release information from GitHub API
  fetchLatestReleaseInfo();

  // Add smooth scrolling for navigation
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      e.preventDefault();

      document.querySelector(this.getAttribute("href")).scrollIntoView({
        behavior: "smooth",
      });
    });
  });

  // Add animation on scroll for features
  const featureRows = document.querySelectorAll(".feature-row");

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        }
      });
    },
    {
      threshold: 0.2,
    }
  );

  featureRows.forEach((row) => {
    observer.observe(row);
  });
});

async function fetchLatestReleaseInfo() {
  try {
    const response = await fetch(
      "https://api.github.com/repos/buda-base/tibetan-ocr-app/releases/latest"
    );
    const releaseData = await response.json();

    if (releaseData.assets && releaseData.assets.length > 0) {
      // Find download links for each platform
      const windowsAsset = releaseData.assets.find(
        (asset) => asset.name.includes("windows") && asset.name.includes("x64")
      );

      const macSiliconAsset = releaseData.assets.find(
        (asset) => asset.name.includes("macos") && asset.name.includes("arm64")
      );

      const macIntelAsset = releaseData.assets.find(
        (asset) => asset.name.includes("macos") && asset.name.includes("x64")
      );

      // Update download links if assets are found
      if (windowsAsset) {
        document.getElementById("windows-download").href =
          windowsAsset.browser_download_url;
      }

      if (macSiliconAsset) {
        document.getElementById("mac-silicon-download").href =
          macSiliconAsset.browser_download_url;
      }

      if (macIntelAsset) {
        document.getElementById("mac-intel-download").href =
          macIntelAsset.browser_download_url;
      }
    }
  } catch (error) {
    console.error("Error fetching release information:", error);
    // Fallback to the hardcoded URLs in the HTML if there's an error
  }
}
