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
    previewArea.innerHTML = ''; // clear existing previews

    if (input.files) {
        var files = Array.from(input.files);

        files.forEach(file => {
            var reader = new FileReader();

            reader.onload = function(e) {
                var img = document.createElement('img');
                img.src = e.target.result;
                img.width = 100; // set width (you can adjust as needed)
                previewArea.appendChild(img);
            }

            reader.readAsDataURL(file);
        });

        document.getElementById('fileName').textContent = files.map(file => file.name).join(', ');
    }
}


// 이미지 업로드 후 OCR 처리
$("#imageUploadForm").submit(function(e) {
    e.preventDefault();
    var formData = new FormData(this);
    
    // 로딩 시작
    $("#loading").show();

    $.ajax({
        type: 'POST',
        url: '/myhome/ocr',
        data: formData,
        contentType: false,
        processData: false,
        // 로딩 종료
        success: function(data) {
            console.log(data);
            $("#loading").hide();

            $("#ocrOriginalImage").html('<img src="' + URL.createObjectURL($("#imageFile")[0].files[0]) + '" width="100%">');
            if (data.text) {
                $("#ocrTextResult").text(data.text);
            } else {
                $("#ocrTextResult").text("오류: 서버에서 올바른 데이터 형식이 아닙니다.");
            }
            $('#ocrResultModal').modal('show');

        },
        error: function() {
            // 로딩 종료 및 에러 메시지 표시
            $("#loading").hide();
            alert('OCR 처리 중 오류가 발생했습니다. 다시 시도해주세요.');
        }
    });
});