<?php
// 设置内容类型为 JSON
header('Content-Type: application/json');

// 获取 POST 请求中的数据
$selectedOptions = isset($_POST['selectedOptions']) ? $_POST['selectedOptions'] : null;

// 初始化响应数组
$response = array('status' => 'error');

// 检查是否接收到数据
if ($selectedOptions !== null) {
    $jsonData = json_decode($selectedOptions, true);
    $filePath = './exps/data_' . date('Ymd_His') . '.json';
    if (file_put_contents($filePath, json_encode($jsonData))) {
        // 如果成功，设置响应状态为 success
        $response['status'] = 'success';
        // $response['info'] = $jsonData;
    } else {
        // 如果失败，记录具体错误信息
        $response['message'] = 'Failed to write to file.';
    }
} else {
    $response['message'] = 'No data received.';
}

// 返回 JSON 响应
echo json_encode($response);
