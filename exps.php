<?php
header('Content-Type: application/json');
// 设置默认时区为北京时间
date_default_timezone_set('Asia/Shanghai');

// 根据请求类型执行不同逻辑
if ($_SERVER['REQUEST_METHOD'] === 'GET') {
    // 定义 wav 文件所在的目录
    $directorys = ['./exps/all', './exps/mode', './exps/pitch', './exps/tempo', './exps/none'];
    // GET 请求逻辑
    $wavFiles = array();
    foreach ($directorys as $directory) {
        $wavFiles = array_merge($wavFiles, glob($directory . '/*.wav'));
    }

    $filesArray = array();
    foreach ($wavFiles as $file) {
        if (is_file($file)) {
            $filesArray[] = array('url' => $file);
        }
    }

    // 输出 JSON 响应
    echo json_encode($filesArray);
} elseif ($_SERVER['REQUEST_METHOD'] === 'POST') { // POST 请求逻辑
    // 获取 POST 请求中的数据
    $selectedOptions = isset($_POST['selectedOptions']) ? $_POST['selectedOptions'] : null;

    // 初始化响应数组
    $response = array('status' => 'error');

    // 检查是否接收到数据
    if ($selectedOptions !== null) {
        $jsonData = json_decode($selectedOptions, true);
        $filePath = './exps/data_' . date('Ymd_His') . '.json';
        if (file_put_contents($filePath, json_encode($jsonData, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE))) {
            // 如果成功，设置响应状态为 success
            $response['status'] = 'success';
            // $response['info'] = $jsonData; // 如果需要，可以取消注释
        } else {
            // 如果失败，记录具体错误信息
            $response['message'] = 'Failed to write to file.';
        }
    } else {
        $response['message'] = 'No data received.';
    }

    // 返回 JSON 响应
    echo json_encode($response);
} else { // 其他请求方法返回错误
    http_response_code(405); // 方法不被允许
    echo json_encode(array('status' => 'error', 'message' => 'Method Not Allowed'));
}
