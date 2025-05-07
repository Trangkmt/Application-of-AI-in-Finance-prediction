/**
 * UploadHandler - Handles file upload UI interactions
 * - Shows loading overlay during file upload
 * - Basic file validation
 */
const UploadHandler = {
    // Initialize the upload handler
    init: function() {
        const uploadForm = document.getElementById('uploadForm');
        const fileInput = document.getElementById('file');
        const loadingOverlay = document.getElementById('loadingOverlay');
        const submitBtn = document.getElementById('submitBtn');
        
        if (uploadForm) {
            uploadForm.addEventListener('submit', function(event) {
                // Basic validation
                if (fileInput && !fileInput.value) {
                    event.preventDefault();
                    alert('Vui lòng chọn file trước khi tải lên.');
                    return;
                }
                
                // File extension validation
                if (fileInput && fileInput.files.length > 0) {
                    const fileName = fileInput.files[0].name;
                    const fileExt = fileName.split('.').pop().toLowerCase();
                    
                    if (!['csv', 'xlsx', 'xls'].includes(fileExt)) {
                        event.preventDefault();
                        alert('Chỉ hỗ trợ các file .csv, .xlsx, .xls');
                        return;
                    }
                }
                
                // Show loading overlay
                if (loadingOverlay) {
                    loadingOverlay.style.display = 'flex';
                }
                
                // Disable submit button
                if (submitBtn) {
                    submitBtn.disabled = true;
                    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Đang xử lý...';
                }
            });
        }
        
        // For drag and drop functionality if needed
        this.setupDragAndDrop();
    },
    
    // Setup drag and drop functionality
    setupDragAndDrop: function() {
        const dropZone = document.getElementById('upload');
        const fileInput = document.getElementById('file');
        
        if (dropZone && fileInput) {
            // Prevent default drag behaviors
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, preventDefaults, false);
                document.body.addEventListener(eventName, preventDefaults, false);
            });
            
            // Highlight drop zone when item is dragged over it
            ['dragenter', 'dragover'].forEach(eventName => {
                dropZone.addEventListener(eventName, highlight, false);
            });
            
            // Unhighlight drop zone when item is dragged out
            ['dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, unhighlight, false);
            });
            
            // Handle dropped files
            dropZone.addEventListener('drop', handleDrop, false);
            
            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }
            
            function highlight() {
                dropZone.classList.add('border-primary');
                dropZone.classList.add('bg-light');
            }
            
            function unhighlight() {
                dropZone.classList.remove('border-primary');
                dropZone.classList.remove('bg-light');
            }
            
            function handleDrop(e) {
                const dt = e.dataTransfer;
                const files = dt.files;
                
                if (files.length > 0) {
                    fileInput.files = files;
                    // Show file name visually if needed
                    const fileNameDisplay = document.querySelector('.file-name');
                    if (fileNameDisplay) {
                        fileNameDisplay.textContent = files[0].name;
                    }
                }
            }
        }
    }
};

// Initialize the upload handler when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    UploadHandler.init();
});
