use utils::address::GlobalAddress;
use utils::Nality;

fn main() -> () {
    let n = Nality::new_empty();

    println!("{:?}", n);

    let n2 = n.clone();

    println!("{:?}", n2);

    n2.update(0.3, GlobalAddress::into(585585));

    println!("{:?}", n2);

    let n3 = Nality::new(0.6, GlobalAddress::into(14));

    println!("{:?}", n3);

    println!("Should not be 60129542144");
    println!("{:?}", n3.get().load(std::sync::atomic::Ordering::Relaxed));
}
