<html>
<body>
<?php
    $p = $_POST['p'];
    $q = $_POST['q'];
    $x = $_POST['x'];
    $y = $_POST['y'];
    $output = shell_exec("hyperbolic-demos/tilings2d.py $p $q $x $y");
    echo $output;
?>
</body>
</html>


