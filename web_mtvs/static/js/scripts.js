$(document).ready(function() {
    $("#userInput").on('keypress', function(e) {
        if(e.which == 13) {  // 13은 엔터키의 keyCode
            e.preventDefault();  // 기본 엔터키의 이벤트를 방지합니다 (예: 폼 제출)
            sendMessage();
        }
    });
});

function sendMessage() {
    var userMessage = $("#userInput").val();
    $("#chatbox").append('<div class="chat-message user-message">User: ' + userMessage + '</div>');
    $.post("/myhome/chatbot", { chat: userMessage }, function(response) {
        $("#chatbox").append('<div class="chat-message bot-message">Chatbot: ' + response + '</div>');
        $("#userInput").val('');  // 입력 필드 초기화
        $("#chatbox").scrollTop($("#chatbox")[0].scrollHeight);  // 채팅 창 아래로 스크롤
    }).fail(function() {
        $("#chatbox").append('<div>Error: 서버 응답 없음</div>');
    });
}
// 파일 이름 표시 함수
function displayFileName(input) {
    var previewArea = document.getElementById('filePreview');
    if (!previewArea) {
        console.error("Element with ID 'filePreview' not found.");
        return;
    }
    // Clear existing previews
    previewArea.innerHTML = '';

    if (input.files) {
        var files = Array.from(input.files);

        files.forEach(file => {
            // For text files like CSV, you might want to read some lines
            // or simply display the file name as a preview

            var text = document.createElement('p');
            text.textContent = "Selected file: " + file.name;
            previewArea.appendChild(text);
        });

        var fileNameElement = document.getElementById('fileName');

        if (fileNameElement) {
            fileNameElement.textContent = files.map(file => file.name).join(', ');
        }
    }
}

// CSV UPLOAD
$(document).ready(function(){
    $("#csvUploadForm").submit(function(e){
        e.preventDefault();
        var formData = new FormData(this);

        $("#loading").show(); // 로딩 표시

        $.ajax({
            url: 'myhome/upload',
            type: 'POST',
            data: formData,
            success: function (response) {
                alert(response.message);
                $("#loading").hide(); // 로딩 숨김

                // if(response.status === 'success') {
                //     window.location.href = 'Map.html'; // 혹은 Flask에서 지정한 라우트로 이동
                //     // console.log("success");
                // }
            },
            cache: false,
            contentType: false,
            processData: false
        });
    });
});