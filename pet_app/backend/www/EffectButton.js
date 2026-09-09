//担当:A

let activesubBox = null;

//ボタン展開用
function opensub(name){
    const newsubBox = document.getElementById("sub_" + name);
    if (activesubBox && activesubBox !== newsubBox) {
        activesubBox.style.display = "none";
    }

    if (activesubBox === newsubBox){
        newsubBox.style.display = "none";
        activesubBox = null;
        return;
    }

    newsubBox.style.display = "flex";
    activesubBox = newsubBox;
}


function applyeffect(effectName){
    console.log("エフェクト：",effectName);
    handleClick(effectName);
}