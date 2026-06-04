use std::env;

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    match args.as_slice() {
        [first, ..] if first == "status" => {
            println!("OK");
        }
        [first, ..] if first == "selftest" => {
            println!("PASS: connected");
        }
        [first, second, third, ..] if first == "model" && second == "auth" && third == "login" => {
            println!(r#"{{"ok":true,"connection":"pending-connection"}}"#);
        }
        [first, ..] if first == "doctor" => {}
        _ => {}
    }
}
