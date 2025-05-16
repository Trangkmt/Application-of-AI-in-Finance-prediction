/**
 * Dashboard - Handles dashboard-specific functionality
 * - Chart visualizations
 * - Interactive UI elements
 */
const Dashboard = {
    init: function() {
        // Initialize interactive elements
        this.setupTooltips();
        this.setupImagePreview();
        this.setupTabNavigation();
    },
    
    // Initialize Bootstrap tooltips
    setupTooltips: function() {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    },
    
    // Setup image preview on click
    setupImagePreview: function() {
        const previewableImages = document.querySelectorAll('.previewable-image');
        
        previewableImages.forEach(img => {
            img.addEventListener('click', function() {
                const modal = new bootstrap.Modal(document.getElementById('imagePreviewModal'));
                const modalImage = document.getElementById('previewModalImage');
                const modalTitle = document.getElementById('imagePreviewModalLabel');
                
                if (modalImage && modalTitle) {
                    modalImage.src = this.src;
                    modalTitle.textContent = this.alt || 'Image Preview';
                    modal.show();
                }
            });
        });
    },
    
    // Setup tab navigation if present
    setupTabNavigation: function() {
        const tabTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tab"]'));
        tabTriggerList.map(function (tabTriggerEl) {
            tabTriggerEl.addEventListener('shown.bs.tab', function (event) {
                // Store the currently active tab in the URL hash
                window.location.hash = event.target.getAttribute('data-bs-target');
            });
        });
        
        // Activate the tab from URL hash if present
        const hash = window.location.hash;
        if (hash) {
            const tab = document.querySelector(`[data-bs-target="${hash}"]`);
            if (tab) {
                const bsTab = new bootstrap.Tab(tab);
                bsTab.show();
            }
        }
    }
};

// Initialize Dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    Dashboard.init();

    // Handle tab navigation from URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    const tab = urlParams.get('tab');
    
    if (tab) {
        const tabEl = document.querySelector(`#${tab}-tab`);
        if (tabEl) {
            const tabInstance = new bootstrap.Tab(tabEl);
            tabInstance.show();
        }
    }
    
    // Add effects for visualization cards
    const vizCards = document.querySelectorAll('.viz-card');
    vizCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.querySelector('.viz-img-container img').style.transform = 'scale(1.05)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.querySelector('.viz-img-container img').style.transform = 'scale(1)';
        });
    });
});
