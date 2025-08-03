// Ensure sidebar toggle button remains visible and functional
document.addEventListener('DOMContentLoaded', function() {
    const ensureToggleButtonVisible = () => {
        const toggleButton = document.querySelector('[data-testid="stSidebarCollapseButton"]');
        const app = document.querySelector('.stApp');
        if (toggleButton && app) {
            // Force visibility and positioning
            toggleButton.style.display = 'block';
            toggleButton.style.visibility = 'visible';
            toggleButton.style.opacity = '1';
            toggleButton.style.position = 'fixed';
            toggleButton.style.top = '2rem';
            toggleButton.style.zIndex = '10000';
            toggleButton.style.transform = 'none';
            // Dynamically adjust left position based on sidebar state
            toggleButton.style.left = app.dataset.collapsed === 'true' ? '0px' : '16rem';
            // Ensure parent doesn't hide it
            const sidebar = document.querySelector('[data-testid="stSidebar"]');
            if (sidebar) {
                sidebar.style.overflow = 'visible';
            }
        } else {
            console.warn('Toggle button or .stApp not found');
        }
    };

    // Run initially
    ensureToggleButtonVisible();

    // Run on click events
    document.addEventListener('click', function(e) {
        if (e.target.closest('[data-testid="stSidebarCollapseButton"]')) {
            setTimeout(ensureToggleButtonVisible, 100); // Reduced delay for faster response
        }
    });

    // Observe DOM changes on stApp and stSidebar
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'childList' || mutation.type === 'attributes') {
                ensureToggleButtonVisible();
            }
        });
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['data-collapsed', 'style', 'class']
    });

    // More frequent periodic check for Streamlit reruns and full-screen mode
    setInterval(ensureToggleButtonVisible, 500);
});