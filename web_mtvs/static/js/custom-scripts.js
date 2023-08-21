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
            url: '/myhome/upload',
            type: 'POST',
            data: formData,
            success: function (response) {
                alert(response.message);
                $("#loading").hide(); // 로딩 숨김
            },
            cache: false,
            contentType: false,
            processData: false
        });
    });
});


// $("#imageUploadForm").submit(function(e) {
//     e.preventDefault();
//     var formData = new FormData(this);
    
//     // 로딩 시작
//     $("#loading").show();

//     $.ajax({
//         type: 'POST',
//         url: '/mymap/map',
//         data: formData,
//         cache: false,
//         contentType: false,
//         processData: false,
//         success: function(data) {
//             console.log(data);

//             // 로딩 종료
//             $("#loading").hide();

//             if (data.hasOwnProperty('imageFile')) {
//                 $("#ocrOriginalImage").html('<img src="' + URL.createObjectURL($("#imageFile")[0].files[0]) + '" width="100%">');
//             }

//             // categories 정보가 있다면 표시
//             if (data.hasOwnProperty('categories')) {
//                 $("#ocrTextResult").append("<p>Categories: " + data.categories + "</p>");
//             }

//             // text 정보가 있다면 표시
//             if (data.hasOwnProperty('categories_basis')) {
//                 $("#ocrTextResult").append("<p>Text: " + data.categories_basis + "</p>");
//             }

//             // 두 가지 정보 모두 없을 경우
//             if (!data.hasOwnProperty('categories') && !data.hasOwnProperty('categories_basis')) {
//                 $("#ocrTextResult").text("오류: 서버에서 올바른 데이터 형식이 아닙니다.");
//             }

//             $('#ocrResultModal').modal('show');
//         },
//         error: function() {
//             // 로딩 종료 및 에러 메시지 표시
//             $("#loading").hide();
//             alert('OCR 처리 중 오류가 발생했습니다. 다시 시도해주세요.');
//         }
//     });
// });
