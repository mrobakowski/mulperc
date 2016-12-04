use conrod;
use conrod::backend::glium::glium;
use conrod::backend::glium::glium::{DisplayBuild, Surface};
use window_gui;
use std;
use image;
use find_folder;
use img;

const WIN_W: u32 = window_gui::WIN_W;
const WIN_H: u32 = window_gui::WIN_H;

pub fn window_loop() {
    let display = glium::glutin::WindowBuilder::new()
        .with_vsync()
        .with_dimensions(WIN_W, WIN_H)
        .with_title("Mulperc")
        .build_glium()
        .unwrap();

    let mut app = window_gui::AppState::new();
    let mut ui = conrod::UiBuilder::new([WIN_W as f64, WIN_H as f64]).theme(window_gui::theme()).build();
    let ids = window_gui::Ids::new(ui.widget_id_generator());
    let assets = find_folder::Search::KidsThenParents(3, 5).for_folder("assets").unwrap();
    let font_path = assets.join("fonts/NotoSans/NotoSans-Regular.ttf");
    ui.fonts.insert_from_file(font_path).unwrap();

    let mut image_map: conrod::image::Map<glium::Texture2d> = window_gui::image_map(&ids);
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

        if ui.global_input.events().next().is_some() {
            let mut ui = ui.set_widgets();
            window_gui::gui(&mut ui, &ids, &mut app);
        }

        if let Some(primitives) = ui.draw_if_changed() {
            renderer.fill(&display, primitives, &image_map);
            let mut target = display.draw();
            target.clear_color(0.0, 0.0, 0.0, 1.0);
            renderer.draw(&display, &mut target, &image_map).unwrap();
            target.finish().unwrap();
        }

        std::thread::sleep(std::time::Duration::from_millis(16));
    }
}