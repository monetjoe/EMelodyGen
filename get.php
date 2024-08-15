<?php
header('Content-Type: application/json');

$wavFiles = array();
$directorys = ['./exps/all', './exps/mode', './exps/pitch', './exps/tempo'];
foreach ($directorys as $directory) {
    $wavFiles = array_merge($wavFiles, glob($directory . '/*.wav'));
}

$filesArray = array();
foreach ($wavFiles as $file) {
    if (is_file($file)) {
        $filesArray[] = array('url' => $file);
    }
}

echo json_encode($filesArray);
