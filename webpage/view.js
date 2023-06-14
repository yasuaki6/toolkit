const target = document.getElementById('test')
console.log(target)

let = ObserverOption={
    threshold:1
}

const observer = new IntersectionObserver(observerEvent,ObserverOption)

observer.observe(target)

function observerEvent(){
    target.style.opacity=1;
}