/**
 * SessionManager - Handles session state management in the client-side
 * - Retrieves session ID from server or localStorage
 * - Manages the display of session-related navigation elements
 */
const SessionManager = {
    // Key used for localStorage
    STORAGE_KEY: 'finance_analysis_session_id',
    
    /**
     * Initialize the session manager
     */
    init: function() {
        // Check if server provided a session ID
        const serverSessionId = document.getElementById('server-session-id');
        
        if (serverSessionId && serverSessionId.value) {
            // Save the session ID from server
            this.saveSessionId(serverSessionId.value);
        }
        
        // Update UI based on active session
        this.updateUI();
        
        // Add event listeners
        this.setupEventListeners();
    },
    
    /**
     * Set up event listeners related to session management
     */
    setupEventListeners: function() {
        // Example: Log out button to clear session
        const logoutBtn = document.getElementById('clear-session');
        if (logoutBtn) {
            logoutBtn.addEventListener('click', function() {
                SessionManager.clearSession();
                window.location.href = '/';
            });
        }
    },
    
    /**
     * Save session ID to localStorage
     * @param {string} sessionId - The session ID to save
     */
    saveSessionId: function(sessionId) {
        localStorage.setItem(this.STORAGE_KEY, sessionId);
    },
    
    /**
     * Get the current session ID from localStorage
     * @returns {string|null} The session ID or null if not present
     */
    getSessionId: function() {
        return localStorage.getItem(this.STORAGE_KEY);
    },
    
    /**
     * Clear the current session from localStorage
     */
    clearSession: function() {
        localStorage.removeItem(this.STORAGE_KEY);
        this.updateUI();
    },
    
    /**
     * Update UI elements based on the session state
     */
    updateUI: function() {
        const sessionId = this.getSessionId();
        const analysisNav = document.getElementById('analysis-nav');
        const currentSessionBadge = document.getElementById('current-session-badge');
        
        if (sessionId) {
            // Show analysis navigation when there's an active session
            if (analysisNav) {
                analysisNav.classList.remove('d-none');
            }
            
            // Update the session ID badge
            if (currentSessionBadge) {
                currentSessionBadge.textContent = sessionId;
            }
            
            // Add the session ID to all analysis links 
            const analysisLinks = document.querySelectorAll('[data-analysis-link]');
            analysisLinks.forEach(link => {
                const baseUrl = link.getAttribute('data-base-url');
                if (baseUrl) {
                    link.href = baseUrl + sessionId;
                }
            });
        } else {
            // Hide analysis navigation when there's no active session
            if (analysisNav) {
                analysisNav.classList.add('d-none');
            }
        }
    }
};

// Initialize SessionManager when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    SessionManager.init();
});
