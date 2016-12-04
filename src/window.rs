use conrod;
use conrod::backend::glium::glium;
use conrod::backend::glium::glium::{DisplayBuild, Surface};
use window_gui;
use std;
use image;
use find_folder;
use img;
use std::path::Path;
use std::cell::{RefCell, Cell};
use mnist::MnistDigits;

const WIN_W: u32 = window_gui::WIN_W;
const WIN_H: u32 = window_gui::WIN_H;

pub fn window_loop() {
    let display = glium::glutin::WindowBuilder::new()
        .with_vsync()
        .with_dimensions(WIN_W, WIN_H)
        .with_title("Mulperc")
        .build_glium()
        .unwrap();

    let mut app = RefCell::new(window_gui::AppState::new());
    let mut ui = conrod::UiBuilder::new([WIN_W as f64, WIN_H as f64]).theme(window_gui::theme()).build();
    let ids = window_gui::Ids::new(ui.widget_id_generator());
    let assets = find_folder::Search::KidsThenParents(3, 5).for_folder("assets").unwrap();
    let font_path = assets.join("fonts/NotoSans/NotoSans-Regular.ttf");
    let mnist = MnistDigits::default_training_set().unwrap();
    ui.fonts.insert_from_file(font_path).unwrap();

    let mut force_update = Cell::new(false);

    let mut image_map = RefCell::new(window_gui::image_map(
        &ids,
        load_image(&display, app.borrow().classifier.image.path
            .as_ref().map(|s| s.as_str()).unwrap_or("assets/images/rust.png")),
        load_mnist(&display, &mnist[app.borrow().mnist.idx])
    ));

    let mut classifier_img_updater = KeyValueUpdatable::new(|| {
        app.borrow().classifier.image.path
            .as_ref().map(|s| s.to_string()).unwrap_or("assets/images/rust.png".into())
    }, |path| {
        let image = load_image(&display, path);
        image_map.borrow_mut().insert(ids.classifier_preview_img, image);
    });

    let mut mnist_updater = KeyValueUpdatable::new(|| {
        app.borrow().mnist.idx
    }, |idx| {
        let data = &mnist[app.borrow().mnist.idx];
        app.borrow_mut().mnist.label = data.1.clone();
        force_update.set(true);
        image_map.borrow_mut().insert(ids.mnist_img,
            load_mnist(&display, data));
    });

    let mut renderer = conrod::backend::glium::Renderer::new(&display).unwrap();

    'main: loop {
        for event in display.poll_events() {
            let window = display.get_window().unwrap();
            if let Some(event) = conrod::backend::glutin::convert(event.clone(), window) {
                ui.handle_event(event);
            }

            match event {
                glium::glutin::Event::DroppedFile(path) => {
                    // on dropped file
                }

                // Break from the loop upon `Escape`.
                glium::glutin::Event::KeyboardInput(_, _, Some(glium::glutin::VirtualKeyCode::Escape)) |
                glium::glutin::Event::Closed =>
                    break 'main,

                _ => {},
            }
        }

        if let Some(win_rect) = ui.rect_of(ui.window) {
            let (win_w, win_h) = (win_rect.w() as u32, win_rect.h() as u32);
            let (w, h) = display.get_window().unwrap().get_inner_size_points().unwrap();
            if w != win_w || h != win_h {
                let event = conrod::event::Input::Resize(w, h);
                ui.handle_event(event);
            }
        }

        classifier_img_updater.update();
        mnist_updater.update();

        if ui.global_input.events().next().is_some() || force_update.get() {
            let mut ui = ui.set_widgets();
            window_gui::gui(&mut ui, &ids, &mut *app.borrow_mut());
            force_update.set(false);
        }

        if let Some(primitives) = ui.draw_if_changed() {
            renderer.fill(&display, primitives, &*image_map.borrow());
            let mut target = display.draw();
            target.clear_color(0.0, 0.0, 0.0, 1.0);
            renderer.draw(&display, &mut target, &*image_map.borrow()).unwrap();
            target.finish().unwrap();
        }

        std::thread::sleep(std::time::Duration::from_millis(16));
    }
}

fn load_image<P: AsRef<Path>>(display: &glium::Display, path: P) -> glium::texture::Texture2d {
    let rgba_image = image::open(path.as_ref()).unwrap().to_rgba();
    let image_dimensions = rgba_image.dimensions();
    let raw_image = glium::texture::RawImage2d::from_raw_rgba_reversed(rgba_image.into_raw(), image_dimensions);
    let texture = glium::texture::Texture2d::new(display, raw_image).unwrap();
    texture
}

fn load_mnist(display: &glium::Display, datapoint: &(Vec<f64>, String)) -> glium::texture::Texture2d {
    let rgba_image = image::DynamicImage::ImageLuma8(image::ImageBuffer::<image::Luma<u8>, _>::from_raw(28, 28, datapoint.0.iter()
        .map(|f| (f * 255.0) as u8).collect::<Vec<u8>>()).unwrap()).to_rgba();
    let image_dimensions = rgba_image.dimensions();
    let raw_image = glium::texture::RawImage2d::from_raw_rgba_reversed(rgba_image.into_raw(), image_dimensions);
    let texture = glium::texture::Texture2d::new(display, raw_image).unwrap();
    texture
}

pub struct KeyValueUpdatable<K, V, KeyGetter, ValueGetter> {
    old_key: K,
    key_getter: KeyGetter,
    value_getter: ValueGetter,
    value: V
}

impl<K, V, KeyGetter, ValueGetter> KeyValueUpdatable<K, V, KeyGetter, ValueGetter>
where KeyGetter: Fn() -> K,
      ValueGetter: Fn(&K) -> V,
      K: PartialEq {
    fn new(kg: KeyGetter, vg: ValueGetter) -> Self {
        let k = kg();
        let v = vg(&k);
        KeyValueUpdatable {
            old_key: k,
            key_getter: kg,
            value_getter: vg,
            value: v,
        }
    }

    fn update_value_if_key_changed(&mut self) {
        let new_k = (&self.key_getter)();
        if new_k != self.old_key {
            self.value = (&self.value_getter)(&new_k);
            self.old_key = new_k;
        }
    }
}

trait Updatable {
    fn update(&mut self);
}

impl<K, V, KeyGetter, ValueGetter> Updatable for KeyValueUpdatable<K, V, KeyGetter, ValueGetter>
where KeyGetter: Fn() -> K,
      ValueGetter: Fn(&K) -> V,
      K: PartialEq {
    fn update(&mut self) {
        self.update_value_if_key_changed()
    }
}