/**
 * UploadHandler - Xử lý tương tác giao diện người dùng khi tải lên tệp
 * - Hiển thị màn hình tải trong quá trình tải lên tệp
 * - Kiểm tra cơ bản tệp
 */
const UploadHandler = {
    // Khởi tạo trình xử lý tải lên
    init: function() {
        const uploadForm = document.getElementById('uploadForm');
        const fileInput = document.getElementById('file');
        const loadingOverlay = document.getElementById('loadingOverlay');
        const submitBtn = document.getElementById('submitBtn');
        
        if (uploadForm) {
            uploadForm.addEventListener('submit', function(event) {
                // Kiểm tra cơ bản
                if (fileInput && !fileInput.value) {
                    event.preventDefault();
                    alert('Vui lòng chọn file trước khi tải lên.');
                    return;
                }
                
                // Kiểm tra phần mở rộng của tệp
                if (fileInput && fileInput.files.length > 0) {
                    const fileName = fileInput.files[0].name;
                    const fileExt = fileName.split('.').pop().toLowerCase();
                    
                    if (!['csv', 'xlsx', 'xls'].includes(fileExt)) {
                        event.preventDefault();
                        alert('Chỉ hỗ trợ các file .csv, .xlsx, .xls');
                        return;
                    }
                }
                
                // Hiển thị màn hình tải
                if (loadingOverlay) {
                    loadingOverlay.style.display = 'flex';
                }
                
                // Vô hiệu hóa nút gửi
                if (submitBtn) {
                    submitBtn.disabled = true;
                    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Đang xử lý...';
                }
            });
        }
        
        // Cho chức năng kéo và thả nếu cần
        this.setupDragAndDrop();
    },
    
    // Thiết lập chức năng kéo và thả
    setupDragAndDrop: function() {
        const dropZone = document.getElementById('upload');
        const fileInput = document.getElementById('file');
        
        if (dropZone && fileInput) {
            // Ngăn chặn hành vi kéo mặc định
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, preventDefaults, false);
                document.body.addEventListener(eventName, preventDefaults, false);
            });
            
            // Làm nổi bật vùng thả khi kéo mục qua nó
            ['dragenter', 'dragover'].forEach(eventName => {
                dropZone.addEventListener(eventName, highlight, false);
            });
            
            // Bỏ nổi bật vùng thả khi kéo mục ra khỏi nó
            ['dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, unhighlight, false);
            });
            
            // Xử lý các tệp đã thả
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
                    // Hiển thị tên tệp nếu cần
                    const fileNameDisplay = document.querySelector('.file-name');
                    if (fileNameDisplay) {
                        fileNameDisplay.textContent = files[0].name;
                    }
                }
            }
        }
    }
};

// Khởi tạo trình xử lý tải lên khi DOM được tải
document.addEventListener('DOMContentLoaded', function() {
    UploadHandler.init();
});
